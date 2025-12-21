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

PY_EXEC="$SCRATCH/no_paper_env/bin/python"
BENCHMARK_SCRIPT="$SLURM_SUBMIT_DIR/benchmark_v2.py" # Assumes patch is applied

JOB_ID=${SLURM_JOB_ID}
BASE_DIR="${SCRATCH}/neural_ops_paper/experiments_${JOB_ID}"
PROFILE_DIR="${BASE_DIR}/profiles"
mkdir -p "${BASE_DIR}" "${PROFILE_DIR}" "logs"
cd "${BASE_DIR}"

export TORCHINDUCTOR_CACHE_DIR="${BASE_DIR}/torch_cache"
mkdir -p "$TORCHINDUCTOR_CACHE_DIR"

CSV_FILE="${BASE_DIR}/diagnostics_results.csv"
echo "Tag,Model,Dim,Res,Batch,Precision,Latency(ms),Throughput(img/s),Mem(MB),Compiled,DataType" > $CSV_FILE

echo ">>> Starting Diagnostics Run on node $(hostname)"
echo ">>> Results will be saved to: $CSV_FILE"

# --- Utility function for running and logging ---
run_and_log() {
    local TAG=$1
    shift
    echo "Running: $@"
    # Execute the command, capture output, and handle potential errors
    OUTPUT=$($PY_EXEC $BENCHMARK_SCRIPT "$@" 2>&1)
    if [[ $? -ne 0 ]]; then
        echo "ERROR during execution: $@" >> error_log.txt
        echo "$OUTPUT" >> error_log.txt
        echo "ERR,$TAG,${@//--/}" | tr ' ' ',' >> $CSV_FILE
    else
        echo "$OUTPUT" | grep "RESULT," | sed "s/RESULT,/$TAG,/" >> $CSV_FILE
    fi
}

# --- PHASE 1: PRECISION SWEEP (With bfloat16 fix) ---
# Goal: Re-run the original sweep to get complete data for all precisions.
echo "--- Phase 1: Full Precision Sweep (3D TFNO) ---"
for PREC in "fp32" "tf32" "bf16"
do
    for COMPILE_FLAG in "" "--compile"
    do
        run_and_log "Phase1_PrecSweep" --model TFNO --dim 3 --res 64 --batch 4 --width 32 --modes 12 --precision $PREC $COMPILE_FLAG
        run_and_log "Phase1_PrecSweep" --model TFNO --dim 3 --res 32 --batch 8 --width 32 --modes 12 --precision $PREC $COMPILE_FLAG
    done
done

# --- PHASE 2: COMPILE SCALING ANALYSIS ---
# Goal: Find the resolution where torch.compile becomes faster than eager.
echo "--- Phase 2: Compile Overhead Analysis (TFNO 3D, bf16) ---"
for RES in 32 48 64 80 96
do
    # Adjust batch size to avoid OOM at high resolutions
    BATCH=8
    if [ "$RES" -gt "48" ]; then BATCH=4; fi
    if [ "$RES" -gt "80" ]; then BATCH=2; fi

    # Eager run
    run_and_log "Phase2_CompileScale" --model TFNO --dim 3 --res $RES --batch $BATCH --width 32 --modes 12 --precision bf16
    # Compiled run
    run_and_log "Phase2_CompileScale" --model TFNO --dim 3 --res $RES --batch $BATCH --width 32 --modes 12 --precision bf16 --compile
done

# --- PHASE 3: TARGETED NSIGHT SYSTEMS PROFILING ---
# Goal: Visually inspect the CUDA kernels for the most interesting cases.
echo "--- Phase 3: Targeted Nsight Systems Profiling ---"

# Case 1: The "Slowdown" Case. Why is Res 32 compiled slow?
srun nsys profile -t cuda,nvtx -o "${PROFILE_DIR}/trace_3d_res32_bf16_compiled" --force-overwrite true \
    $PY_EXEC $BENCHMARK_SCRIPT \
        --model TFNO --dim 3 --res 32 --batch 8 --precision bf16 --compile --unroll 20

# Case 2: The "Speedup" Case. What does good compilation look like?
srun nsys profile -t cuda,nvtx -o "${PROFILE_DIR}/trace_3d_res64_bf16_compiled" --force-overwrite true \
    $PY_EXEC $BENCHMARK_SCRIPT \
        --model TFNO --dim 3 --res 64 --batch 4 --precision bf16 --compile --unroll 20

# Case 3: The Baseline. Eager mode for comparison.
srun nsys profile -t cuda,nvtx -o "${PROFILE_DIR}/trace_3d_res64_bf16_eager" --force-overwrite true \
    $PY_EXEC $BENCHMARK_SCRIPT \
        --model TFNO --dim 3 --res 64 --batch 4 --precision bf16 --unroll 20


echo ">>> Diagnostics Run Complete."
echo ">>> CSV Data: $CSV_FILE"
echo ">>> Nsys Profiles: $PROFILE_DIR"