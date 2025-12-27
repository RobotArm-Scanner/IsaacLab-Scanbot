#!/usr/bin/env bash
set -eo pipefail

source /opt/conda/etc/profile.d/conda.sh
git clone https://github.com/rise-policy/rise
cd rise
conda create -n rise python=3.9 -y
conda activate rise

export CONDA_BACKUP_CXX=""
conda install -y -c "nvidia/label/cuda-12.8.0" cuda-toolkit
conda install -y -c conda-forge openblas-devel ninja
pip install -r requirements.txt
pip uninstall -y torch torchvision
pip install torch==2.7.0 torchvision --index-url https://download.pytorch.org/whl/cu128

# Build dependencies directory
mkdir -p dependencies
cd dependencies

git clone https://github.com/AzharSindhi/MinkowskiEngineCuda13
cd MinkowskiEngineCuda13

export TORCH_CUDA_ARCH_LIST="7.5 8.0 8.6 9.0"

python setup.py install \
    --blas_include_dirs=${CONDA_PREFIX}/include \
    --blas_library_dirs=${CONDA_PREFIX}/lib \
    --blas=openblas \
    --force_cuda

pip install fvcore
pip install https://github.com/MiroPsota/torch_packages_builder/releases/download/pytorch3d-0.7.8%2B5043d15/pytorch3d-0.7.8%2B5043d15pt2.7.0cu128-cp39-cp39-linux_x86_64.whl