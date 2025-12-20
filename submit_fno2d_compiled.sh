#!/bin/bash -l
#SBATCH --account=m4647
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --time=00:20:00
#SBATCH --job-name=tfno_profile
#SBATCH --output=profile_tfno-%j.out
#SBATCH -C 'gpu&hbm80g'

module load python
module load cudatoolkit

# --- Setup ---
PY_EXEC="$SCRATCH/no_paper_env/bin/python"
JOB_DIR="${SCRATCH}/neural_ops_paper/profile_tfno_${SLURM_JOB_ID}"
mkdir -p ${JOB_DIR}
cd ${JOB_DIR}

export TORCHINDUCTOR_CACHE_DIR="${JOB_DIR}/torch_cache"
export NSYS_TMP_DIR="${JOB_DIR}/nsys_tmp"
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$NSYS_TMP_DIR"

# --- Run Profiler ---
echo "Profiling TFNO 2D Res 256..."

srun nsys profile \
    --trace=cuda,nvtx \
    --capture-range=cudaProfilerApi \
    --output="${JOB_DIR}/tfno_2d_256_compiled" \
    --force-overwrite=true \
    --stats=true \
    $PY_EXEC $SLURM_SUBMIT_DIR/benchmark.py \
        --model FNO \
        --dim 2 \
        --res 256 \
        --batch 16 \
        --modes 16 \
        --width 64 \
        --unroll 50 \
        --data real \
        --compile

echo "Done. File saved to ${JOB_DIR}/tfno_2d_256_compiled.nsys-rep"