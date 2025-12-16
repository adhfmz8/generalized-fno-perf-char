#!/bin/bash -l

#SBATCH --account=m4647
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --job-name=fno_paper_bench
#SBATCH --output=%x-%j.out
#SBATCH -C 'gpu&hbm80g'

module load python

# Setup Env
PY_EXEC="$SCRATCH/no_paper_env/bin/python"
JOB_DIR="${SCRATCH}/neural_ops_paper/job_${SLURM_JOB_ID}"
mkdir -p ${JOB_DIR}
cp ${SLURM_SUBMIT_DIR}/benchmark.py ${JOB_DIR}/
cd ${JOB_DIR}

# Create CSV Header
echo "RESULT,Model,Dim,Res,Batch,Latency(ms),Throughput,Mem(MB),Compiled" > results.csv

# EXPERIMENT 1: 2D Comparative Analysis
# Compare Algorithmic efficiency (FNO vs TFNO vs UNO vs HeavyCNN)
echo "--- Starting Experiment 1: 2D Models ---"

MODELS=("FNO" "TFNO" "UNO" "HEAVYCNN")
# High resolutions to stress the 2D plane
RESOLUTIONS=(128 256 512)
BATCHES=(16 8 4)

for MODEL in "${MODELS[@]}"; do
    for i in "${!RESOLUTIONS[@]}"; do
        RES="${RESOLUTIONS[$i]}"
        BATCH="${BATCHES[$i]}"
        
        # 1. Standard Benchmark (CSV collection)
        $PY_EXEC benchmark.py \
            --model ${MODEL} \
            --dim 2 \
            --res ${RES} \
            --batch ${BATCH} \
            --modes 16 --width 64 --unroll 50 >> results.csv
            
        # 2. Profiling (Optional: Comment out to save time if not analyzing traces yet)
        srun nsys profile --trace=cuda,nvtx --output="${MODEL}_2D_r${RES}" --force-overwrite=true \
            $PY_EXEC benchmark.py --model ${MODEL} --dim 2 --res ${RES} --batch ${BATCH} --unroll 10 > /dev/null
    done
done

# ==========================================
# EXPERIMENT 2: 3D Hardware Limits
# Stressing Memory Capacity (FNO vs HeavyCNN)
# ==========================================
echo "--- Starting Experiment 2: 3D Models ---"

# Only running FNO (Memory Bound) and HeavyCNN (Compute Bound)
# TFNO is optional here, but skipped for simplicity as requested.
MODELS_3D=("FNO" "HEAVYCNN")

# 3D Resolutions scale cubically. 64^3 = 262k points. 128^3 = 2M points.
RESOLUTIONS_3D=(32 64 96) 
BATCHES_3D=(4 2 1)

for MODEL in "${MODELS_3D[@]}"; do
    for i in "${!RESOLUTIONS_3D[@]}"; do
        RES="${RESOLUTIONS_3D[$i]}"
        BATCH="${BATCHES_3D[$i]}"
        
        $PY_EXEC benchmark.py \
            --model ${MODEL} \
            --dim 3 \
            --res ${RES} \
            --batch ${BATCH} \
            --modes 16 --width 64 --unroll 20 >> results.csv
        
        # 2. Profiling Run (LOW unroll for clean visualization)
        # We only need 3 steps: One to see the start, one steady state, one end.
        echo "Profiling 3D ${MODEL}..."
        srun nsys profile \
            --trace=cuda,nvtx,osrt \
            --output="${MODEL}_3D_r${RES}_profile" \
            --force-overwrite=true \
            --export=sqlite \
            $PY_EXEC benchmark.py \
                --model ${MODEL} \
                --dim 3 \
                --res ${RES} \
                --batch ${BATCH} \
                --modes 16 --width 64 \
                --unroll 3 > /dev/null
    done
done

echo "Done. Results saved to ${JOB_DIR}/results.csv"