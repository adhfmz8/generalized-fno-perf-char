#!/bin/bash -l
#SBATCH --account=m4647
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=01:30:00
#SBATCH --job-name=fno_ncu
#SBATCH --output=logs/ncu_profile-%j.out
#SBATCH --error=logs/ncu_profile-%j.err
#SBATCH -C 'gpu&hbm80g'

# --- 1. Environment and Setup ---
module load python
module load cudatoolkit

# Path Setup (Adjust if necessary)
PY_EXEC="$SCRATCH/no_paper_env/bin/python"
BENCHMARK_SCRIPT="$SLURM_SUBMIT_DIR/benchmark_v2.py"

JOB_ID=${SLURM_JOB_ID}
BASE_DIR="${SCRATCH}/neural_ops_paper/experiments_${JOB_ID}"
PROFILE_DIR="${BASE_DIR}/ncu_profiles"
mkdir -p "${BASE_DIR}" "${PROFILE_DIR}" "logs"
cd "${BASE_DIR}"

export TORCHINDUCTOR_CACHE_DIR="${BASE_DIR}/torch_cache"
mkdir -p "$TORCHINDUCTOR_CACHE_DIR"

echo ">>> Starting Nsight Compute Profiling on node $(hostname)"
echo ">>> Saving reports to: $PROFILE_DIR"

# --- 2. Profiling Configuration ---
# We use a small unroll count (5) because NCU replays the kernel many times.
# High unroll counts would make the job take hours.
COMMON_ARGS="--model TFNO --dim 3 --res 64 --batch 4 --precision bf16 --compile --unroll 5"

# --- TARGET 1: The FFT Kernel (Efficiency & Strides) ---
# We use regex to catch the specific cuFFT kernel name seen in your nsys trace.
# --set full: Captures ALL sections (occupancy, memory, compute, source).
echo "--- Profiling Target 1: cuFFT Kernel ---"

dcgmi profile --pause

srun ncu --target-processes all \
    --kernel-name-base function \
    --kernel-regex "regular_fft" \
    --launch-count 1 \
    --set full \
    --force-overwrite \
    --outfile "${PROFILE_DIR}/ncu_fft_report" \
    $PY_EXEC $BENCHMARK_SCRIPT $COMMON_ARGS

if [ $? -eq 0 ]; then
    echo ">>> FFT Profile Complete: ${PROFILE_DIR}/ncu_fft_report.ncu-rep"
else
    echo ">>> FFT Profile FAILED."
fi

# --- TARGET 2: The Triton Fused Kernel (Memory Bandwidth) ---
# We target 'triton_' to see how efficient the torch.compile kernels are.
# We focus on SpeedOfLight and Memory to see if we are bandwidth bound.
echo "--- Profiling Target 2: Triton Fused Kernels ---"

srun ncu --target-processes all \
    --kernel-name-base function \
    --kernel-regex "triton_" \
    --launch-count 1 \
    --section SpeedOfLight --section MemoryWorkloadAnalysis --section Occupancy \
    --force-overwrite \
    --outfile "${PROFILE_DIR}/ncu_triton_report" \
    $PY_EXEC $BENCHMARK_SCRIPT $COMMON_ARGS

if [ $? -eq 0 ]; then
    echo ">>> Triton Profile Complete: ${PROFILE_DIR}/ncu_triton_report.ncu-rep"
else
    echo ">>> Triton Profile FAILED."
fi

dcgmi profile --resume

echo ">>> Nsight Compute Run Complete."