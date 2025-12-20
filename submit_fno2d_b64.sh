#!/bin/bash -l
#SBATCH --account=m4647
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --time=00:20:00
#SBATCH --job-name=fno_b64
#SBATCH --output=profile_fno_b64-%j.out
#SBATCH -C 'gpu&hbm80g'

module load python
module load cudatoolkit

# --- Configuration ---
MODEL="FNO"
DIM=2
RES=256
BATCH=64
WIDTH=128
MODES=16

# Construct a unique string for filenames
# Format: MODEL_DIM_RES_BATCH_WIDTH_MODES
RUN_TAG="${MODEL}_${DIM}d_res${RES}_b${BATCH}_w${WIDTH}_m${MODES}_compiled"

# --- Setup ---
PY_EXEC="$SCRATCH/no_paper_env/bin/python"
JOB_DIR="${SCRATCH}/neural_ops_paper/profile_${RUN_TAG}_${SLURM_JOB_ID}"
mkdir -p ${JOB_DIR}
cd ${JOB_DIR}

export TORCHINDUCTOR_CACHE_DIR="${JOB_DIR}/torch_cache"
export NSYS_TMP_DIR="${JOB_DIR}/nsys_tmp"
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$NSYS_TMP_DIR"

# --- Run Profiler ---
echo "Profiling ${MODEL} ${DIM}D Res ${RES} Batch ${BATCH} Width ${WIDTH} Modes ${MODES}..."

srun nsys profile \
    --trace=cuda,nvtx \
    --capture-range=cudaProfilerApi \
    --output="${JOB_DIR}/${RUN_TAG}" \
    --force-overwrite=true \
    --stats=true \
    $PY_EXEC $SLURM_SUBMIT_DIR/benchmark.py \
        --model $MODEL \
        --dim $DIM \
        --res $RES \
        --batch $BATCH \
        --modes $MODES \
        --width $WIDTH \
        --unroll 50 \
        --data real \
        --compile

echo "Done. File saved to ${JOB_DIR}/${RUN_TAG}.nsys-rep"