#!/usr/bin/env bash
set -eo pipefail

source /opt/conda/etc/profile.d/conda.sh
git clone https://github.com/rise-policy/rise
cd rise
conda create -n rise python=3.8 -y
conda activate rise

conda install -y -c "nvidia/label/cuda-11.7.0" cuda-toolkit
conda install -y -c conda-forge openblas-devel ninja
pip install -r requirements.txt

# Build dependencies directory
mkdir -p dependencies
cd dependencies

git clone https://github.com/chenxi-wang/MinkowskiEngine
cd MinkowskiEngine

export TORCH_CUDA_ARCH_LIST="7.5 8.0 8.6"

python setup.py install \
    --blas_include_dirs=${CONDA_PREFIX}/include \
    --blas_library_dirs=${CONDA_PREFIX}/lib \
    --blas=openblas \
    --force_cuda

pip install fvcore
pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py38_cu113_pyt1110/download.html