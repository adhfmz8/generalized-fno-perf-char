#!/bin/bash -l
#SBATCH --account=m4647
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=32
#SBATCH --time=01:00:00
#SBATCH --job-name=tfno_eager_check
#SBATCH --output=logs/tfno_eager-%j.out
#SBATCH --error=logs/tfno_eager-%j.err
#SBATCH -C 'gpu&hbm80g'

# --- Environment Setup ---
module load python
module load cudatoolkit

# Use the same python executable as your previous runs
PY_EXEC="$SCRATCH/no_paper_env/bin/python"

# Directory Setup (Creating a specific 'debug' folder for this comparison)
BASE_DIR="${SCRATCH}/neural_ops_paper/tfno_debug_${SLURM_JOB_ID}"
PROFILE_DIR="${BASE_DIR}/profiles"
mkdir -p "${BASE_DIR}" "${PROFILE_DIR}" "logs"
cd "${BASE_DIR}"

echo ">>> Starting TFNO Eager Mode Check on node $(hostname)"
echo ">>> Profiles will be saved to: $PROFILE_DIR"

# --- Run: TFNO 3D (Eager / Uncompiled) ---
# Goal: Check if 'Eager' mode avoids the graph breaks and overhead seen in 'Compiled' mode.

srun nsys profile \
    --trace=cuda,nvtx,osrt \
    --capture-range=cudaProfilerApi \
    --output="${PROFILE_DIR}/hero_tfno_3d_64_eager" \
    --force-overwrite=true \
    --stats=true \
    $PY_EXEC $SLURM_SUBMIT_DIR/benchmark.py \
        --model TFNO \
        --dim 3 \
        --res 64 \
        --batch 2 \
        --width 32 \
        --modes 12 \
        --unroll 20 \
        --data synthetic \


echo ">>> Run Complete."