#!/bin/bash -l
#SBATCH --account=m4647
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=02:00:00
#SBATCH --job-name=fno_diagnostics
#SBATCH --output=logs/diag_sweep-%j.out
#SBATCH --error=logs/diag_sweep-%j.err
#SBATCH -C 'gpu&hbm80g'

# --- 1. Environment and Setup ---
module load python
module load cudatoolkit

# Update this path to your actual environment
PY_EXEC="$SCRATCH/no_paper_env/bin/python"
BENCHMARK_SCRIPT="$SLURM_SUBMIT_DIR/benchmark_v2.py"

JOB_ID=${SLURM_JOB_ID}
BASE_DIR="${SCRATCH}/neural_ops_paper/experiments_${JOB_ID}"
PROFILE_DIR="${BASE_DIR}/profiles"
mkdir -p "${BASE_DIR}" "${PROFILE_DIR}" "logs"
cd "${BASE_DIR}"

export TORCHINDUCTOR_CACHE_DIR="${BASE_DIR}/torch_cache"
mkdir -p "$TORCHINDUCTOR_CACHE_DIR"

# Enable Graph Break logging to verify compilation
export TORCH_LOGS="graph_breaks"

CSV_FILE="${BASE_DIR}/diagnostics_results.csv"
# Header matches print statement in python script
echo "Tag,Model,Dim,Res,Batch,Precision,Latency(ms),Throughput(img/s),Mem(MB),Compiled,DataType,Params(M),GFLOPs_Fwd,TFLOPs_Eff" > $CSV_FILE

echo ">>> Starting Diagnostics Run on node $(hostname)"
echo ">>> Results will be saved to: $CSV_FILE"

# --- Utility function for running and logging ---
run_and_log() {
    local TAG=$1
    shift
    echo "Running: $@"
    # Execute the command, capture output
    OUTPUT=$($PY_EXEC $BENCHMARK_SCRIPT "$@" 2>&1)
    if [[ $? -ne 0 ]]; then
        echo "ERROR during execution: $@" >> error_log.txt
        echo "$OUTPUT" >> error_log.txt
        echo "ERR,$TAG,${@//--/}" | tr ' ' ',' >> $CSV_FILE
    else
        # Extract the RESULT line
        echo "$OUTPUT" | grep "RESULT," | sed "s/RESULT,/$TAG,/" >> $CSV_FILE
    fi
}

# --- PHASE 1: PRECISION SWEEP ---
echo "--- Phase 1: Full Precision Sweep (3D TFNO) ---"
for PREC in "fp32" "tf32" "bf16"
do
    for COMPILE_FLAG in "" "--compile"
    do
        # Larger Resolution
        run_and_log "Phase1_PrecSweep" --model TFNO --dim 3 --res 64 --batch 4 --width 32 --modes 12 --precision $PREC $COMPILE_FLAG
        # Smaller Resolution
        run_and_log "Phase1_PrecSweep" --model TFNO --dim 3 --res 32 --batch 8 --width 32 --modes 12 --precision $PREC $COMPILE_FLAG
    done
done

# --- PHASE 2: COMPILE SCALING ANALYSIS ---
echo "--- Phase 2: Compile Overhead Analysis (TFNO 3D, bf16) ---"
for RES in 32 48 64 80 96
do
    BATCH=8
    if [ "$RES" -gt "48" ]; then BATCH=4; fi
    if [ "$RES" -gt "80" ]; then BATCH=2; fi

    run_and_log "Phase2_CompileScale" --model TFNO --dim 3 --res $RES --batch $BATCH --width 32 --modes 12 --precision bf16
    run_and_log "Phase2_CompileScale" --model TFNO --dim 3 --res $RES --batch $BATCH --width 32 --modes 12 --precision bf16 --compile
done

# --- PHASE 3: TARGETED NSIGHT SYSTEMS PROFILING ---
echo "--- Phase 3: Targeted Nsight Systems Profiling ---"
# Nsight Systems Flags Explanation:
# -t cuda,nvtx,osrt,cudnn,cublas: Trace CUDA kernels, NVTX markers, OS Runtime (sched), and Libraries
# -s cpu: Sample CPU backtraces to find Python overhead (critical for small batches)
# --capture-range=cudaProfilerApi: Only profile the 'profiler.start()' region (skips warmup)

NSYS_CMD="nsys profile -t cuda,nvtx,osrt,cudnn,cublas -s cpu --capture-range=cudaProfilerApi --stop-on-range-end=true --force-overwrite true"

# 1. Clean up old profiles
rm -f "${PROFILE_DIR}"/*.nsys-rep
rm -f "${PROFILE_DIR}"/*.qdstrm

# Case 1: Compiled 3D small res (Likely CPU bound or overhead bound)
srun -n 1 --ntasks-per-node=1 $NSYS_CMD -o "${PROFILE_DIR}/trace_3d_res32_bf16_compiled" \
    $PY_EXEC $BENCHMARK_SCRIPT \
        --model TFNO --dim 3 --res 32 --batch 8 --precision bf16 --compile --unroll 20

# Case 2: Compiled 3D large res (Compute bound)
srun -n 1 --ntasks-per-node=1 $NSYS_CMD -o "${PROFILE_DIR}/trace_3d_res64_bf16_compiled" \
    $PY_EXEC $BENCHMARK_SCRIPT \
        --model TFNO --dim 3 --res 64 --batch 4 --precision bf16 --compile --unroll 20

# Case 3: Eager Baseline (To visualize the kernel fragmentation vs compiled)
srun -n 1 --ntasks-per-node=1 $NSYS_CMD -o "${PROFILE_DIR}/trace_3d_res64_bf16_eager" \
    $PY_EXEC $BENCHMARK_SCRIPT \
        --model TFNO --dim 3 --res 64 --batch 4 --precision bf16 --unroll 20

echo ">>> Diagnostics Run Complete."
echo ">>> CSV Data: $CSV_FILE"
echo ">>> Nsys Profiles: $PROFILE_DIR"