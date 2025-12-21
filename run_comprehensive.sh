#!/bin/bash -l
#SBATCH --account=m4647
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=32
#SBATCH --time=02:00:00
#SBATCH --job-name=fno_comprehensive
#SBATCH --output=logs/comp_sweep-%j.out
#SBATCH --error=logs/comp_sweep-%j.err
#SBATCH -C 'gpu&hbm80g'

# --- 1. Environment Setup ---
module load python
module load cudatoolkit

# Point to your python
PY_EXEC="$SCRATCH/no_paper_env/bin/python"

# Directories
JOB_ID=${SLURM_JOB_ID}
BASE_DIR="${SCRATCH}/neural_ops_paper/experiments_${JOB_ID}"
PROFILE_DIR="${BASE_DIR}/profiles"
mkdir -p "${BASE_DIR}" "${PROFILE_DIR}" "logs"
cd "${BASE_DIR}"

# Torch Compilation Cache
export TORCHINDUCTOR_CACHE_DIR="${BASE_DIR}/torch_cache"
export NSYS_TMP_DIR="${BASE_DIR}/nsys_tmp"
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$NSYS_TMP_DIR"

CSV_FILE="${BASE_DIR}/comprehensive_results.csv"
echo "Tag,Model,Dim,Res,Batch,Precision,Latency(ms),Throughput(img/s),Mem(MB),Compiled,DataType" > $CSV_FILE

echo ">>> Starting Comprehensive Run on node $(hostname)"
echo ">>> Results: $CSV_FILE"

# PHASE 1: PRECISION & SCALING SWEEP (Latency/Throughput Only)
# Goal: Quantify the speedup of BF16 (Tensor Cores) vs FP32 vs TF32
#       and verify scaling laws across Dim 2 and 3.

echo "--- Phase 1: 3D Scaling & Precision Sweep ---"

# We focus on 3D TFNO as it was your bottleneck. 
# We run 3 Precisions: FP32 (Baseline), TF32 (A100 Native), BF16 (AMP)
# We run Uncompiled vs Compiled (Default)

for PREC in "fp32" "tf32" "bf16"
do
    for COMPILE_FLAG in "" "--compile"
    do
        # 3D Scaling (Bottleneck check)
        # Using a slightly higher unroll for stability
        echo "Running TFNO 3D [Prec:$PREC] [Compile:$COMPILE_FLAG]..."
        
        # Res 64
        OUTPUT=$($PY_EXEC $SLURM_SUBMIT_DIR/benchmark_v2.py \
            --model TFNO --dim 3 --res 64 --batch 4 --width 32 --modes 12 \
            --unroll 50 --data synthetic --precision $PREC $COMPILE_FLAG)
        echo "$OUTPUT" | grep "RESULT," | sed "s/RESULT,/Phase1_3D_Prec,/" >> $CSV_FILE
        
        # Res 32 (Small)
        OUTPUT=$($PY_EXEC $SLURM_SUBMIT_DIR/benchmark_v2.py \
            --model TFNO --dim 3 --res 32 --batch 8 --width 32 --modes 12 \
            --unroll 50 --data synthetic --precision $PREC $COMPILE_FLAG)
        echo "$OUTPUT" | grep "RESULT," | sed "s/RESULT,/Phase1_3D_Prec,/" >> $CSV_FILE
    done
done

# PHASE 2: HERO PROFILING (Nsight Systems)
# Goal: Capture traces to visualize the "Indexing" kernel and memory gaps.
# We focus on the BEST performing configuration (BF16 + Compile) vs the Worst (FP32 Eager).

echo "--- Phase 2: Hero Profiling ---"

# 1. Baseline: TFNO 3D, Res 64, Eager, FP32 (The "Old" way)
# This serves as the reference for "how bad it was".
srun nsys profile \
    --trace=cuda,nvtx,osrt \
    --capture-range=cudaProfilerApi \
    --output="${PROFILE_DIR}/trace_tfno_3d_fp32_eager" \
    --force-overwrite=true \
    --stats=true \
    $PY_EXEC $SLURM_SUBMIT_DIR/benchmark_v2.py \
        --model TFNO --dim 3 --res 64 --batch 4 --width 32 --modes 12 \
        --unroll 20 --data synthetic --precision fp32

# 2. Modern: TFNO 3D, Res 64, Compiled, BF16 (The "New" way)
# This checks if the "Index Elementwise" kernel disappears or speeds up with AMP.
srun nsys profile \
    --trace=cuda,nvtx,osrt \
    --capture-range=cudaProfilerApi \
    --output="${PROFILE_DIR}/trace_tfno_3d_bf16_compiled" \
    --force-overwrite=true \
    --stats=true \
    $PY_EXEC $SLURM_SUBMIT_DIR/benchmark_v2.py \
        --model TFNO --dim 3 --res 64 --batch 4 --width 32 --modes 12 \
        --unroll 20 --data synthetic --precision bf16 --compile

# 3. High-Res 2D FNO Check (Res 512) - Compiled + TF32
# Just to ensure 2D scaling is healthy on A100s
srun nsys profile \
    --trace=cuda,nvtx,osrt \
    --capture-range=cudaProfilerApi \
    --output="${PROFILE_DIR}/trace_fno_2d_512_tf32_compiled" \
    --force-overwrite=true \
    --stats=true \
    $PY_EXEC $SLURM_SUBMIT_DIR/benchmark_v2.py \
        --model FNO --dim 2 --res 512 --batch 16 --width 64 --modes 24 \
        --unroll 20 --data synthetic --precision tf32 --compile

echo ">>> All Runs Complete."
echo ">>> CSV Data: $CSV_FILE"
echo ">>> Profiles: $PROFILE_DIR"