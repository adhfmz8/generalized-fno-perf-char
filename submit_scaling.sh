#!/bin/bash -l

#SBATCH --account=m4647
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=none
#SBATCH --time=04:00:00
#SBATCH --job-name=no_paper_scaling
#SBATCH --output=%x-%j.out
#SBATCH --mail-user=nkdiamond@miners.utep.edu
#SBATCH --mail-type=ALL
#SBATCH -C 'gpu&hbm80g'

module load python

# Define the specific Python executable from your Env
PY_EXEC="$SCRATCH/no_paper_env/bin/python"

# debug
echo "Using Python interpreter: $PY_EXEC"
$PY_EXEC -c "import neuralop; print('NeuralOp imported successfully from', neuralop.__file__)"

# Setup Scratch Workspace
JOB_DIR="${SCRATCH}/neural_ops_paper/job_${SLURM_JOB_ID}"
mkdir -p ${JOB_DIR}

# Copy benchmark file
cp ${SLURM_SUBMIT_DIR}/benchmark.py ${JOB_DIR}/
cd ${JOB_DIR}

echo "Working directory: ${JOB_DIR}"

# Define Experiment Matrix
MODELS=("FNO" "TFNO" "UNO") 
RESOLUTIONS=(64 128 256)
BATCH_SIZES=(16 8 4)

# Create a CSV header file
echo "RESULT,Model,Res,Batch,Modes,Width,Latency(ms),Throughput,Mem(MB)" > ${JOB_DIR}/results.csv

for MODEL in "${MODELS[@]}"; do
    for i in "${!RESOLUTIONS[@]}"; do
        RES="${RESOLUTIONS[$i]}"
        BATCH="${BATCH_SIZES[$i]}"
        REPORT_NAME="${MODEL}_r${RES}_b${BATCH}"
        
        echo "Processing: $MODEL | Res: $RES | Batch: $BATCH"
        
        # 1. LATENCY RUN (No Profiler) - Accurate CSV metrics
        # We append output to the CSV file
        $PY_EXEC benchmark.py \
            --model ${MODEL} \
            --res ${RES} \
            --batch ${BATCH} \
            --modes 16 \
            --unroll 100 \
            | grep "RESULT" >> ${JOB_DIR}/results.csv

        # 2. PROFILING RUN (With Nsys) - For visual analysis
        # Reduced unroll count to save disk space on traces
        srun nsys profile \
            --trace=cuda,nvtx,osrt,cudnn,cublas \
            --sample=cpu \
            --output="${REPORT_NAME}" \
            --force-overwrite=true \
            $PY_EXEC benchmark.py \
                --model ${MODEL} \
                --res ${RES} \
                --batch ${BATCH} \
                --modes 16 \
                --unroll 20 > /dev/null
    done
done