#!/bin/bash -l
#SBATCH --account=m4647
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=32
#SBATCH --time=04:00:00
#SBATCH --job-name=fno_paper_sweep
#SBATCH --output=logs/paper_sweep-%j.out
#SBATCH --error=logs/paper_sweep-%j.err
#SBATCH -C 'gpu&hbm80g'

# --- 1. Environment Setup ---
module load python
module load cudatoolkit

# Point to your specific python environment
PY_EXEC="$SCRATCH/no_paper_env/bin/python"

# Directory management
JOB_ID=${SLURM_JOB_ID}
BASE_DIR="${SCRATCH}/neural_ops_paper/experiments_${JOB_ID}"
PROFILE_DIR="${BASE_DIR}/profiles"
mkdir -p "${BASE_DIR}" "${PROFILE_DIR}" "logs"
cd "${BASE_DIR}"

# Torch Compilation Cache (prevent recompiling every run)
export TORCHINDUCTOR_CACHE_DIR="${BASE_DIR}/torch_cache"
export NSYS_TMP_DIR="${BASE_DIR}/nsys_tmp"
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$NSYS_TMP_DIR"

# CSV Output File
CSV_FILE="${BASE_DIR}/sweep_results.csv"
echo "Tag,Model,Dim,Res,Batch,Latency(ms),Throughput(img/s),Mem(MB),Compiled,DataType" > $CSV_FILE

echo ">>> Starting Paper Experiments on node $(hostname)"
echo ">>> Outputting results to $CSV_FILE"

# ==============================================================================
# PHASE 1: STATISTICAL SWEEP (Raw Metrics for Graphs)
# ==============================================================================
# We run these WITHOUT nsys to get pure performance numbers.
# We sweep: Resolution, Model Type, Compilation.
# ==============================================================================

# 1. 2D Resolution Scaling (Scaling Laws)
# Models: FNO vs TFNO vs HeavyCNN
# Res: 128, 256, 512, 1024 (1024 checks OOM behavior)
# Width: Fixed at 64 for fairness, Modes: 16
echo "--- Starting Phase 1: 2D Resolution Scaling ---"

for MODEL in "FNO" "TFNO" "HeavyCNN"
do
    for RES in 128 256 512 1024
    do
        # Skip HeavyCNN on 1024 if it's too slow (optional check)
        
        # Run Uncompiled
        echo "Running $MODEL 2D Res $RES (Eager)..."
        OUTPUT=$($PY_EXEC $SLURM_SUBMIT_DIR/benchmark.py \
            --model $MODEL --dim 2 --res $RES --batch 16 --width 64 --modes 16 \
            --unroll 100 --data synthetic)
        
        # Parse the RESULT line and append to CSV
        echo "$OUTPUT" | grep "RESULT," | sed "s/RESULT,/Phase1_ResScale,/" >> $CSV_FILE

        # Run Compiled
        echo "Running $MODEL 2D Res $RES (Compiled)..."
        OUTPUT=$($PY_EXEC $SLURM_SUBMIT_DIR/benchmark.py \
            --model $MODEL --dim 2 --res $RES --batch 16 --width 64 --modes 16 \
            --unroll 100 --data synthetic --compile)
        
        echo "$OUTPUT" | grep "RESULT," | sed "s/RESULT,/Phase1_ResScale,/" >> $CSV_FILE
    done
done

# 2. 3D Scaling (The Heavy Compute Case)
# Models: FNO vs TFNO (UNO disabled for 3D in your script)
echo "--- Starting Phase 1: 3D Scaling ---"

for MODEL in "FNO" "TFNO"
do
    for RES in 32 64 128
    do
         # Lower batch size for 3D to fit in memory
        BATCH=4
        if [ "$RES" -eq 128 ]; then BATCH=1; fi

        echo "Running $MODEL 3D Res $RES (Compiled)..."
        OUTPUT=$($PY_EXEC $SLURM_SUBMIT_DIR/benchmark.py \
            --model $MODEL --dim 3 --res $RES --batch $BATCH --width 32 --modes 12 \
            --unroll 50 --data synthetic --compile)
        
        echo "$OUTPUT" | grep "RESULT," | sed "s/RESULT,/Phase1_3DScale,/" >> $CSV_FILE
    done
done

# ==============================================================================
# PHASE 2: "HERO" RUNS WITH PROFILING (Nsight Systems)
# ==============================================================================
# We only profile specific high-interest cases to generate timelines.
# Use 'real' data here to see if I/O or irregularity impacts kernels.
# ==============================================================================

echo "--- Starting Phase 2: Nsight Profiling ---"

# Case A: High-Res 2D FNO (Analyze FFT vs Conv cost)
srun nsys profile \
    --trace=cuda,nvtx,osrt \
    --capture-range=cudaProfilerApi \
    --output="${PROFILE_DIR}/hero_fno_2d_512_compiled" \
    --force-overwrite=true \
    --stats=true \
    $PY_EXEC $SLURM_SUBMIT_DIR/benchmark.py \
        --model FNO --dim 2 --res 512 --batch 8 --width 64 --modes 24 \
        --unroll 20 --data synthetic --compile

# Case B: TFNO 3D (Analyze Tensor Contraction efficiency)
srun nsys profile \
    --trace=cuda,nvtx,osrt \
    --capture-range=cudaProfilerApi \
    --output="${PROFILE_DIR}/hero_tfno_3d_64_compiled" \
    --force-overwrite=true \
    --stats=true \
    $PY_EXEC $SLURM_SUBMIT_DIR/benchmark.py \
        --model TFNO --dim 3 --res 64 --batch 2 --width 32 --modes 12 \
        --unroll 20 --data synthetic --compile

echo ">>> Experiments Complete."
echo ">>> CSV Data: $CSV_FILE"
echo ">>> Profiles: $PROFILE_DIR"