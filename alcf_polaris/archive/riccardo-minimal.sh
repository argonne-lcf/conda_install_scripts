#!/bin/bash

# New conda env
module use /soft/modulefiles
module load conda/2024-04-29
conda activate
conda create -y -p /lus/eagle/projects/datascience/balin/Nek/GNN/env/gnn_nccl --solver=libmamba --clone base
conda activate /lus/eagle/projects/datascience/balin/Nek/GNN/env/gnn_nccl

# Uninstall ML packages from conda env
pip uninstall -y torch
pip uninstall -y torch-sparse torch_spline_conv


# Install PyTorch from source
export PT_REPO_TAG="v2.3.0"
export PT_REPO_URL=https://github.com/pytorch/pytorch.git
export MPICH_GPU_SUPPORT_ENABLED=1

export CUDA_VERSION_MAJOR=12
export CUDA_VERSION_MINOR=4
export CUDA_VERSION_MINI=1

export CUDA_VERSION=$CUDA_VERSION_MAJOR.$CUDA_VERSION_MINOR
export CUDA_VERSION_FULL=$CUDA_VERSION.$CUDA_VERSION_MINI

export CUDA_TOOLKIT_BASE=/soft/compilers/cudatoolkit/cuda-${CUDA_VERSION_FULL}
export CUDA_HOME=${CUDA_TOOLKIT_BASE}

export CUDA_DEPS_BASE=/soft/libraries/

export CUDNN_VERSION_MAJOR=9
export CUDNN_VERSION_MINOR=1
export CUDNN_VERSION_EXTRA=0.70
export CUDNN_VERSION=$CUDNN_VERSION_MAJOR.$CUDNN_VERSION_MINOR.$CUDNN_VERSION_EXTRA
export CUDNN_BASE=$CUDA_DEPS_BASE/cudnn/cudnn-cuda$CUDA_VERSION_MAJOR-linux-x64-v$CUDNN_VERSION

export NCCL_VERSION_MAJOR=2
export NCCL_VERSION_MINOR=21.5-1
export NCCL_VERSION=$NCCL_VERSION_MAJOR.$NCCL_VERSION_MINOR
export NCCL_BASE=$CUDA_DEPS_BASE/nccl/nccl_$NCCL_VERSION+cuda${CUDA_VERSION}_x86_64

export TENSORRT_VERSION_MAJOR=8
export TENSORRT_VERSION_MINOR=6.1.6
export TENSORRT_VERSION=$TENSORRT_VERSION_MAJOR.$TENSORRT_VERSION_MINOR
export TENSORRT_BASE=$CUDA_DEPS_BASE/trt/TensorRT-$TENSORRT_VERSION.Linux.x86_64-gnu.cuda-12.0

git clone $PT_REPO_URL
cd pytorch
if [[ -z "$PT_REPO_TAG" ]]; then
    echo "Checkout PyTorch master"
else
    echo "Checkout PyTorch tag $PT_REPO_TAG"
    git checkout --recurse-submodules $PT_REPO_TAG
    echo "git submodule sync"
    git submodule sync
    echo "git submodule update"
    git submodule update --init --recursive
fi

export CUDNN_INCLUDE_DIR=$CUDNN_BASE/include
export CPATH="$CPATH:$CUDNN_INCLUDE_DIR"


export CRAY_ACCEL_TARGET="nvidia80"
export CRAY_TCMALLOC_MEMFS_FORCE="1"
export CRAYPE_LINK_TYPE="dynamic"
export CRAY_ACCEL_VENDOR="nvidia"
export CRAY_CPU_TARGET="x86-64"

export USE_CUDA=1
export USE_CUDNN=1
export TORCH_CUDA_ARCH_LIST=8.0
export CUDNN_ROOT=$CUDNN_BASE
echo "CUDNN_ROOT=$CUDNN_BASE"
export CUDNN_ROOT_DIR=$CUDNN_BASE
export CUDNN_INCLUDE_DIR=$CUDNN_BASE/include
export NCCL_ROOT_DIR=$NCCL_BASE

export USE_TENSORRT=ON
export TENSORRT_ROOT=$TENSORRT_BASE
export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}

export PYTORCH_BUILD_VERSION="${PT_REPO_TAG:1}"
export PYTORCH_BUILD_NUMBER=1

export TENSORRT_INCLUDE_DIR="TENSORRT_BASE/include"
export TENSORRT_LIBRARY="$TENSORRT_BASE/lib/libmyelin.so"

export USE_CUSPARSELT=1
export CUSPARSELT_ROOT="/soft/libraries/cusparselt/libcusparse_lt-linux-x86_64-0.6.0.6/"
export CUSPARSELT_INCLUDE_PATH="/soft/libraries/cusparselt/libcusparse_lt-linux-x86_64-0.6.0.6/include"

export USE_SYSTEM_NCCL=1
export NCCL_ROOT=$NCCL_BASE
export NCCL_INCLUDE_DIR=$NCCL_BASE/include
export NCCL_LIB_DIR=$NCCL_BASE/lib

BUILD_TEST=0 CUDAHOSTCXX=g++-12 CC=/usr/bin/gcc-12 CXX=/usr/bin/g++-12 python setup.py bdist_wheel
PT_WHEEL=$(find dist/ -name "torch*.whl" -type f)
echo "pip installing $PT_WHEEL"
pip install $PT_WHEEL
cd ..

# Torch_Cluster
CC=cc pip install torch_cluster==1.6.3

# Install other packages
CC=cc pip install mpipartition
