# Setup env

## 1. Load the base python module to get 'mamba' (faster conda)
`module load python`

## 2. Create an empty environment in your scratch directory
 We use python 3.10 as it's very stable for current PyTorch
`conda create -p $SCRATCH/no_paper_env python=3.10 -y`

## 3. Activate it
source activate $SCRATCH/no_paper_env

## 4. Install PyTorch with CUDA 12 support (Required for A100s)
`pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121`

## 5. Install your specific libraries
`pip install neuraloperator nvtx configmypy tensorly tensorly-torch matplotlib pandas`

## 6. Verify it works (should print 'True')
`python -c "import torch; print(torch.cuda.is_available())"`