#!/bin/bash -l
#SBATCH --account=m4647
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=02:00:00
#SBATCH --job-name=mem_stride
#SBATCH --output=check_strides.out
#SBATCH -C 'gpu&hbm80g'

module load python

PY_EXEC="$SCRATCH/no_paper_env/bin/python"
$PY_EXEC check_strides.py