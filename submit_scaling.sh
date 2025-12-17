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

# Ensure scratch exists for data download
mkdir -p $SCRATCH/neuralop_data

# Create CSV Header
echo "RESULT,Model,Dim,Res,Batch,Latency(ms),Throughput,Mem(MB),Compiled,Data" > results.csv

# ==========================================
# EXPERIMENT 1: 2D Comparative Analysis
# Real Data (Darcy Flow)
# ==========================================
echo "--- Starting Experiment 1: 2D Models (Darcy Flow) ---"

MODELS=("FNO" "TFNO" "UNO" "HEAVYCNN")
# Darcy is natively small, but FNO scales. We will test upscaling.
RESOLUTIONS=(85 128 256) 
BATCHES=(32 16 8)

for MODEL in "${MODELS[@]}"; do
    for i in "${!RESOLUTIONS[@]}"; do
        RES="${RESOLUTIONS[$i]}"
        BATCH="${BATCHES[$i]}"
        
        # Run with Real Data
        $PY_EXEC benchmark.py \
            --model ${MODEL} \
            --dim 2 \
            --res ${RES} \
            --batch ${BATCH} \
            --modes 16 --width 64 --unroll 50 \
            --data real >> results.csv
            
        # Profiling Run
        srun nsys profile --trace=cuda,nvtx --output="${MODEL}_2D_r${RES}" --force-overwrite=true \
            $PY_EXEC benchmark.py --model ${MODEL} --dim 2 --res ${RES} --batch ${BATCH} --unroll 10 --data real > /dev/null
    done
done

# ==========================================
# EXPERIMENT 2: 3D Hardware Limits
# Synthetic Data (Navier Stokes 3D is too large to download in job)
# ==========================================
echo "--- Starting Experiment 2: 3D Models (Synthetic) ---"

MODELS_3D=("FNO" "HEAVYCNN")
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
            --modes 16 --width 64 --unroll 20 \
            --data synthetic >> results.csv
        
        # Profiling 3D
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
                --unroll 3 \
                --data synthetic > /dev/null
    done
done

echo "Done. Results saved to ${JOB_DIR}/results.csv"