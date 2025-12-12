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

# Load the base python module (needed to get conda/mamba func)
module load python

# Activate env
source activate $SCRATCH/no_paper_env

echo "Environment activated: $(which python)"

# Setup Scratch Workspace
JOB_DIR="${SCRATCH}/neural_ops_paper/job_${SLURM_JOB_ID}"
mkdir -p ${JOB_DIR}

# Copy the python script there to ensure we capture the code version used
cp ${SLURM_SUBMIT_DIR}/benchmark.py ${JOB_DIR}/
cd ${JOB_DIR}

echo "Working directory: ${JOB_DIR}"

# Define Experiment Matrix
MODELS=("FNO") # Add "CNO" "UNO" here
# Pair Resolutions with Batch Sizes (Index 0: Res=64, Batch=16)
RESOLUTIONS=(64 128 256)
BATCH_SIZES=(16 8 4)

# DCGM Pause (Best practice on Perlmutter to reduce overhead noise)
echo "Pausing DCGM for clean profiling..."
dcgmi profile --pause

# Run the Sweep
for MODEL in "${MODELS[@]}"; do
    for i in "${!RESOLUTIONS[@]}"; do
        RES="${RESOLUTIONS[$i]}"
        BATCH="${BATCH_SIZES[$i]}"
        
        REPORT_NAME="${MODEL}_r${RES}_b${BATCH}"
        
        echo "------------------------------------------------"
        echo "Running: $MODEL | Res: $RES | Batch: $BATCH"
        echo "------------------------------------------------"

        # NSYS Command
        # -t cuda,nvtx,osrt,cudnn,cublas : Capture everything relevant
        # --capture-range=cudaProfilerApi : Only capture the part inside NVTX (reduces file size)
        # --cuda-graph-trace=node : If you use Cuda Graphs later, this is essential
        
        srun nsys profile \
            --trace=cuda,nvtx,osrt,cudnn,cublas \
            --stats=true \
            --force-overwrite=true \
            --output="${REPORT_NAME}" \
            python benchmark.py \
                --model ${MODEL} \
                --res ${RES} \
                --batch ${BATCH} \
                --modes 16 \
                --unroll 50
    done
done

# Cleanup
echo "Resuming DCGM..."
dcgmi profile --resume

echo "------------------------------------------------"
echo "Job Complete."
echo "Results and Profiles are located in: ${JOB_DIR}"
echo "------------------------------------------------"