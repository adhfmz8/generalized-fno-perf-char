#!/bin/bash
#SBATCH --time=00:05:00
#SBATCH -C gpu
#SBATCH --gpus=1
#SBATCH --output=stride_check.out

module load python

PY_EXEC="$SCRATCH/no_paper_env/bin/python"
$PY_EXEC check_strides.py