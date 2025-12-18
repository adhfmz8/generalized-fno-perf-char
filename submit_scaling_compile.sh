#!/bin/bash -l
#SBATCH --account=m4647
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --job-name=fno_compiled
#SBATCH --output=compile_bench-%j.out
#SBATCH -C 'gpu&hbm80g'

module load python
module load cudatoolkit

# 1. Setup Environment
PY_EXEC="$SCRATCH/no_paper_env/bin/python"
JOB_DIR="${SCRATCH}/neural_ops_paper/compiled_${SLURM_JOB_ID}"
mkdir -p ${JOB_DIR}
cd ${JOB_DIR}

export TORCHINDUCTOR_CACHE_DIR="${JOB_DIR}/torch_cache"
mkdir -p $TORCHINDUCTOR_CACHE_DIR

# 3. Create Results File
echo "RESULT,Model,Dim,Res,Batch,Latency(ms),Throughput,Mem(MB),Compiled,Data" > results_compiled.csv

# ==========================================
# RUN 2D MODELS (Compiled)
# ==========================================
MODELS_2D=("FNO" "TFNO" "UNO" "HEAVYCNN")
RESOLUTIONS_2D=(128 256) 
BATCH_2D=16

for MODEL in "${MODELS_2D[@]}"; do
    for RES in "${RESOLUTIONS_2D[@]}"; do
        echo "Running Compiled ${MODEL} 2D at Res ${RES}..."
        $PY_EXEC $SLURM_SUBMIT_DIR/benchmark.py \
            --model ${MODEL} --dim 2 --res ${RES} --batch ${BATCH_2D} \
            --modes 16 --width 64 --unroll 50 --data real --compile >> results_compiled.csv
    done

    echo "Profiling ${MODEL} at Res ${RES}..."
    srun nsys profile \
        --trace=cuda,nvtx \
        --capture-range=nvtx \
        --nvtx-capture="PROFILE_BLOCK" \
        --output=${JOB_DIR}/profile_${MODEL}_2D_res${RES}" \
        --force-overwrite=true \
        --wait=all \
        $PY_EXEC $SLURM_SUBMIT_DIR/benchmark.py \
            --model ${MODEL} --dim 2 --res ${RES} --batch ${BATCH_2D} \
            --compile --unroll 5 --data real
done

# ==========================================
# RUN 3D MODELS (Compiled)
# ==========================================
MODELS_3D=("FNO" "HEAVYCNN")
RESOLUTIONS_3D=(32 64) 
BATCH_3D=4

for MODEL in "${MODELS_3D[@]}"; do
    for RES in "${RESOLUTIONS_3D[@]}"; do
        echo "Running Compiled ${MODEL} 3D at Res ${RES}..."
        $PY_EXEC $SLURM_SUBMIT_DIR/benchmark.py \
            --model ${MODEL} --dim 3 --res ${RES} --batch ${BATCH_3D} \
            --modes 16 --width 64 --unroll 20 --data synthetic --compile >> results_compiled.csv
    done

    echo "Profiling ${MODEL} at Res ${RES}..."
    srun nsys profile \
        --trace=cuda,nvtx \
        --capture-range=nvtx \
        --nvtx-capture="PROFILE_BLOCK" \
        --output=${JOB_DIR}/profile_${MODEL}_3D_res${RES}" \
        --force-overwrite=true \
        --wait=all \
        $PY_EXEC $SLURM_SUBMIT_DIR/benchmark.py \
            --model ${MODEL} --dim 3 --res ${RES} --batch ${BATCH_3D} \
            --compile --unroll 5 --data synthetic
done

echo "Compiled benchmarking complete. Results in ${JOB_DIR}/results_compiled.csv"