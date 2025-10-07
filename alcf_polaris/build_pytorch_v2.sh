#!/bin/bash -l

# As of May 2022
# This script will install TensorFlow, PyTorch, and Horovod on Polaris, all from source
# 1 - Login to Polaris login-node
# 2 - Run './<this script> /path/to/install/base/'
# 3 - script installs everything down in /path/to/install/base/
# 4 - wait for it to complete

# KGF: check HARDCODE points for lines that potentially require manual edits to pinned package versions

BASE_PATH=$1
DATE_PATH="$(basename $BASE_PATH)"

export PYTHONNOUSERSITE=1
umask 0022

# move primary conda packages directory/cache away from ~/.conda/pkgs (4.2 GB currently)
# hardlinks should be preserved even if these files are moved (not across filesystem boundaries)
export CONDA_PKGS_DIRS=/soft/applications/conda/pkgs

#########################################################
# Check for outside communication
# (be sure not to inherit these vars from dotfiles)
#########################################################
unset https_proxy
unset http_proxy

wget -q --spider -T 10 http://google.com
if [ $? -eq 0 ]; then
    echo "Network Online"
else
    echo "Network Offline, setting proxy envs"
    export https_proxy=http://proxy.alcf.anl.gov:3128
    export http_proxy=http://proxy.alcf.anl.gov:3128
fi

module list
module load PrgEnv-nvidia  # not actually using NVHPC compilers to build TF
#module load PrgEnv-gnu
module load gcc-native-mixed/14.2 # get 14.2.0 (Aug 2024) instead of /usr/bin/gcc 7.5 (2019)
module load cuda
module load craype-accel-nvidia80  # wont load for PrgEnv-gnu; see HPE Case 5367752190
module unload cuda
module load craype-x86-milan
export MPICH_GPU_SUPPORT_ENABLED=1
module unload xalt
module list
echo $MPICH_DIR

# -------------------- begin HARDCODE of major built-from-source frameworks etc.
# unset *_TAG variables to build latest master/main branch (or "develop" in the case of DeepHyper)
#DH_REPO_TAG="0.4.2"
DH_REPO_URL=https://github.com/deephyper/deephyper.git

# KGF: build master for CUDA 12.9.1 compatibility
#TF_REPO_TAG="v2.20.0"  # 2025-08-13
# try CUDA 12.8.1 with hermetic build of TF 2.20.1
TF_REPO_TAG="dcbbe2c058ea3ed206972dcd96345f8f6460eef1"
# 5103439efa9801d4b8b1460b915437b27959c09b from the same day fails
#TF_REPO_TAG="v2.17.1"   # 2024-10-24
PT_REPO_TAG="v2.8.0"
#HOROVOD_REPO_TAG="v0.28.1"
HOROVOD_REPO_TAG=""
TF_REPO_URL=https://github.com/tensorflow/tensorflow.git
HOROVOD_REPO_URL=https://github.com/uber/horovod.git
PT_REPO_URL=https://github.com/pytorch/pytorch.git

############################
# Manual version checks/changes below that must be made compatible with TF/Torch/CUDA versions above:
# - pytorch vision
# - magma-cuda
# - tensorflow_probability
# - torch-geometric, torch-sparse, torch-scatter, pyg-lib
# - cupy
# - jax
###########################

#################################################
# CUDA path and version information
#################################################

CUDA_VERSION_MAJOR=12
CUDA_VERSION_MINOR=9
CUDA_VERSION_MINI=1

CUDA_VERSION=$CUDA_VERSION_MAJOR.$CUDA_VERSION_MINOR
CUDA_VERSION_FULL=$CUDA_VERSION.$CUDA_VERSION_MINI
# KGF: this might be a problem, does not match syntax of NCCL_VERSION, etc.

CUDA_TOOLKIT_BASE=/soft/compilers/cudatoolkit/cuda-${CUDA_VERSION_FULL}
CUDA_HOME=${CUDA_TOOLKIT_BASE}

CUDA_DEPS_BASE=/soft/libraries/

CUDNN_VERSION_MAJOR=9
CUDNN_VERSION_MINOR=13.0
CUDNN_VERSION_EXTRA=50
CUDNN_VERSION=$CUDNN_VERSION_MAJOR.$CUDNN_VERSION_MINOR.$CUDNN_VERSION_EXTRA

# HARDCODE: manually renaming default cuDNN tarball name to fit this schema:
CUDNN_BASE=$CUDA_DEPS_BASE/cudnn/cudnn-cuda$CUDA_VERSION_MAJOR-linux-x64-v$CUDNN_VERSION

NCCL_VERSION_MAJOR=2
NCCL_VERSION_MINOR=28.3-1
NCCL_VERSION=$NCCL_VERSION_MAJOR.$NCCL_VERSION_MINOR
NCCL_BASE=$CUDA_DEPS_BASE/nccl/nccl_$NCCL_VERSION+cuda${CUDA_VERSION}_x86_64

TENSORRT_VERSION_MAJOR=10
TENSORRT_VERSION_MINOR=13.3.9
# TENSORRT_VERSION_MAJOR=8
# TENSORRT_VERSION_MINOR=6.1.6
TENSORRT_VERSION=$TENSORRT_VERSION_MAJOR.$TENSORRT_VERSION_MINOR
# HARDCODE
TENSORRT_BASE=$CUDA_DEPS_BASE/trt/TensorRT-$TENSORRT_VERSION.Linux.x86_64-gnu.cuda-12.9
echo "TENSORRT_BASE=${TENSORRT_BASE}"

#################################################
# TensorFlow Config flags (for ./configure run)
#################################################
export TF_CUDA_COMPUTE_CAPABILITIES=8.0
# Note that TF_CUDA_VERSION and TF_CUDNN_VERSION should consist of major and minor versions only (e.g. 12.3 for CUDA and 9.1 for CUDNN).
# https://openxla.org/xla/hermetic_cuda
export TF_CUDA_VERSION=${CUDA_VERSION_MAJOR}.${CUDA_VERSION_MINOR}
export TF_CUDNN_VERSION=${CUDNN_VERSION_MAJOR}.${CUDNN_VERSION_MINOR}
export TF_TENSORRT_VERSION=$TENSORRT_VERSION_MAJOR
export TF_NCCL_VERSION=${NCCL_VERSION_MAJOR}.${NCCL_VERSION_MINOR}
# KGF: double check above changes to syntax
export CUDA_TOOLKIT_PATH=$CUDA_TOOLKIT_BASE
export CUDNN_INSTALL_PATH=$CUDNN_BASE
export NCCL_INSTALL_PATH=$NCCL_BASE
export TENSORRT_INSTALL_PATH=$TENSORRT_BASE
export TF_NEED_OPENCL_SYCL=0
export TF_NEED_COMPUTECPP=0
export TF_CUDA_CLANG=0   # KGF?
export TF_NEED_OPENCL=0
export TF_NEED_MPI=0
export TF_NEED_ROCM=0
export TF_NEED_CUDA=1
export TF_NEED_TENSORRT=0
export TF_CUDA_PATHS=$CUDA_TOOLKIT_BASE,$CUDNN_BASE,$NCCL_BASE
# KGF: last env var lets 12.9.0 leak in when 12.9.1 is specified, etc.
#export TF_CUDA_PATHS=$CUDA_TOOLKIT_BASE,$CUDNN_BASE,$NCCL_BASE,$TENSORRT_BASE
#export GCC_HOST_COMPILER_PATH=$(which gcc)

# HARDCODE
export TF_PYTHON_VERSION=3.12
# KGF: check this
export GCC_HOST_COMPILER_PATH=/usr/bin/gcc-14
#/opt/cray/pe/gcc/12.2.0/snos/bin/gcc
export CC_OPT_FLAGS="-march=native -Wno-sign-compare"
export TF_SET_ANDROID_WORKSPACE=0

#################################################
## Installing Miniconda
#################################################

# set Conda installation folder and where downloaded content will stay
CONDA_PREFIX_PATH=$BASE_PATH/mconda3
DOWNLOAD_PATH=$BASE_PATH/DOWNLOADS
WHEELS_PATH=$BASE_PATH/wheels

mkdir -p $CONDA_PREFIX_PATH
mkdir -p $DOWNLOAD_PATH
mkdir -p $WHEELS_PATH
cd $BASE_PATH
# HARDCODE
# Download and install conda for a base python installation
# CONDAVER='py312_25.7.0-2'
# CONDA_DOWNLOAD_URL=https://repo.continuum.io/miniconda
# CONDA_INSTALL_SH=Miniconda3-$CONDAVER-Linux-x86_64.sh
#echo "Downloading miniconda installer"
echo "Downloading miniforge installer"
CONDA_DOWNLOAD_URL="https://github.com/conda-forge/miniforge/releases/latest/download"
CONDA_INSTALL_SH="Miniforge3-$(uname)-$(uname -m).sh"
wget $CONDA_DOWNLOAD_URL/$CONDA_INSTALL_SH -P $DOWNLOAD_PATH
chmod +x $DOWNLOAD_PATH/$CONDA_INSTALL_SH

echo "Installing Miniconda"
echo "bash $DOWNLOAD_PATH/$CONDA_INSTALL_SH -b -p $CONDA_PREFIX_PATH -u"
bash $DOWNLOAD_PATH/$CONDA_INSTALL_SH -b -p $CONDA_PREFIX_PATH -u

cd $CONDA_PREFIX_PATH

#########
# create a setup file
cat > setup.sh << EOF
preferred_shell=\$(basename \$SHELL)

module load PrgEnv-gnu

if [ -n "\$ZSH_EVAL_CONTEXT" ]; then
    DIR=\$( cd "\$( dirname "\$0" )" && pwd )
else  # bash, sh, etc.
    DIR=\$( cd "\$( dirname "\${BASH_SOURCE[0]}" )" && pwd )
fi

eval "\$(\$DIR/bin/conda shell.\${preferred_shell} hook)"

# test network
unset https_proxy
unset http_proxy
wget -q --spider -T 10 http://google.com
if [ \$? -eq 0 ]; then
    echo "Network Online"
else
   echo "Network Offline, setting proxy envs"
   export https_proxy=http://proxy.alcf.anl.gov:3128
   export http_proxy=http://proxy.alcf.anl.gov:3128
fi

export CUDA_TOOLKIT_BASE=$CUDA_TOOLKIT_BASE
export CUDNN_BASE=$CUDNN_BASE
export NCCL_BASE=$NCCL_BASE
export TENSORRT_BASE=$TENSORRT_BASE
export LD_LIBRARY_PATH=\$CUDA_TOOLKIT_BASE/lib64:\$CUDNN_BASE/lib:\$NCCL_BASE/lib:\$TENSORRT_BASE/lib:\$LD_LIBRARY_PATH:
export PATH=\$CUDA_TOOLKIT_BASE/bin:\$PATH
EOF

PYTHON_VER=$(ls -d lib/python?.?? | grep -oP '(?<=python)\d+\.\d+')
echo PYTHON_VER=$PYTHON_VER

cat > .condarc << EOF
channels:
   - conda-forge
   - pytorch
env_prompt: "(${DATE_PATH}/{default_env}) "
pkgs_dirs:
   - ${CONDA_PKGS_DIRS}
   - \$HOME/.conda/pkgs
EOF

# move to base install directory
cd $BASE_PATH
echo "cd $BASE_PATH"

# setup conda environment
source $CONDA_PREFIX_PATH/setup.sh
echo "after sourcing conda"
module unload xalt

echo "CONDA BINARY: $(which conda)"
echo "CONDA VERSION: $(conda --version)"
echo "PYTHON VERSION: $(python --version)"

set -e

################################################
### Install TensorFlow
################################################

echo "Conda install some dependencies"
conda install -y -n base conda-libmamba-solver
conda config --set solver libmamba

conda install -y -c conda-forge cmake zip unzip astunparse setuptools future six requests dataclasses graphviz numba numpy pymongo conda-build pip libaio
conda install -y -c conda-forge mkl mkl-include  # onednn mkl-dnn git-lfs ### on ThetaGPU

# CUDA only: Add LAPACK support for the GPU if needed
# HARDCODE
conda install -y -c pytorch -c conda-forge magma-cuda126 # ${CUDA_VERSION_MAJOR}${CUDA_VERSION_MINOR}
conda install -y -c conda-forge mamba ccache

# echo "Clone TensorFlow"
# cd $BASE_PATH
# git clone $TF_REPO_URL
# cd tensorflow

# if [[ -z "$TF_REPO_TAG" ]]; then
#     echo "Checkout TensorFlow master"
# else
#     echo "Checkout TensorFlow tag $TF_REPO_TAG"
#     git checkout --recurse-submodules $TF_REPO_TAG
# fi
# BAZEL_VERSION=$(head -n1 .bazelversion)
# echo "Found TensorFlow depends on Bazel version $BAZEL_VERSION"

# cd $BASE_PATH
# echo "Download Bazel binaries"
# BAZEL_DOWNLOAD_URL=https://github.com/bazelbuild/bazel/releases/download/$BAZEL_VERSION
# BAZEL_INSTALL_SH=bazel-$BAZEL_VERSION-installer-linux-x86_64.sh
# BAZEL_INSTALL_PATH=$BASE_PATH/bazel-$BAZEL_VERSION
# echo "wget $BAZEL_DOWNLOAD_URL/$BAZEL_INSTALL_SH -P $DOWNLOAD_PATH"
# wget $BAZEL_DOWNLOAD_URL/$BAZEL_INSTALL_SH -P $DOWNLOAD_PATH
# chmod +x $DOWNLOAD_PATH/$BAZEL_INSTALL_SH
# echo "Install Bazel in $BAZEL_INSTALL_PATH"
# bash $DOWNLOAD_PATH/$BAZEL_INSTALL_SH --prefix=$BAZEL_INSTALL_PATH
# export PATH=$PATH:/$BAZEL_INSTALL_PATH/bin

cd $BASE_PATH

echo "Install TensorFlow Dependencies"
pip install -U numpy numba ninja
pip install -U pip wheel mock gast portpicker pydot packaging pyyaml
pip install -U keras_applications --no-deps
pip install -U keras_preprocessing --no-deps

# echo "Configure TensorFlow"
# cd tensorflow
export PYTHON_BIN_PATH=$(which python)
export PYTHON_LIB_PATH=$(python -c 'import site; print(site.getsitepackages()[0])')
export TMP=/tmp
# ./configure

# echo "Bazel Build TensorFlow"

# HARDCODE
module use /soft/modulefiles
module load llvm/release-18.1.6  # llvm/release-19.1.7
export CC=/soft/compilers/llvm/release-18.1.6/bin/clang  # TF 2.20.0 tested with Clang 18.1.8
export BAZEL_COMPILER=$CC

# 2.17 and later:
unset CC

#################################################
### Install PyTorch
#################################################

cd $BASE_PATH
echo "Clone PyTorch"

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
# HARDCODE
export CUDNN_INCLUDE_DIR=$CUDNN_BASE/include
export CPATH="$CPATH:$CUDNN_INCLUDE_DIR"

echo "Install PyTorch"
#module unload gcc-mixed
module load PrgEnv-gnu
export CRAY_ACCEL_TARGET="nvidia80"
export CRAY_TCMALLOC_MEMFS_FORCE="1"
export CRAYPE_LINK_TYPE="dynamic"
export CRAY_ACCEL_VENDOR="nvidia"
export CRAY_CPU_TARGET="x86-64"

module unload xalt
module list
echo "CRAY_ACCEL_TARGET= $CRAY_ACCEL_TARGET"
echo "CRAYPE_LINK_TYPE = $CRAYPE_LINK_TYPE"

export USE_CUDA=1
export USE_CUDNN=1
export TORCH_CUDA_ARCH_LIST="8.0"
echo "CUDNN_ROOT=$CUDNN_BASE"
export CUDNN_ROOT_DIR=$CUDNN_BASE
export CUDNN_INCLUDE_DIR=$CUDNN_BASE/include

export USE_SYSTEM_NCCL=1
export NCCL_ROOT=$NCCL_BASE
export NCCL_INCLUDE_DIR=$NCCL_BASE/include
export NCCL_LIB_DIR=$NCCL_BASE/lib

export USE_TENSORRT=ON
export TENSORRT_ROOT=$TENSORRT_BASE
export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}

export PYTORCH_BUILD_VERSION="${PT_REPO_TAG:1}"
export PYTORCH_BUILD_NUMBER=1

export TENSORRT_INCLUDE_DIR="TENSORRT_BASE/include"
export TENSORRT_LIBRARY="$TENSORRT_BASE/lib/libmyelin.so"

export USE_CUSPARSELT=1
# HARDCODE
export CUSPARSELT_ROOT="/soft/libraries/cusparselt/libcusparse_lt-linux-x86_64-0.8.1.1_cuda12-archive/"
export CUSPARSELT_INCLUDE_PATH="${CUSPARSELT_ROOT}include"
# -------------

echo "TENSORRT_INCLUDE_DIR=${TENSORRT_INCLUDE_DIR}"
echo "TENSORRT_LIBRARY=${TENSORRT_LIBRARY}"
echo "CUSPARSELT_ROOT=${CUSPARSELT_ROOT}"
echo "CUSPARSELT_INCLUDE_PATH=${CUSPARSELT_INCLUDE_PATH}"

echo "PYTORCH_BUILD_VERSION=$PYTORCH_BUILD_VERSION and PYTORCH_BUILD_NUMBER=$PYTORCH_BUILD_NUMBER"
#echo "CC=/opt/cray/pe/gcc/12.2.0/snos/bin/gcc CXX=/opt/cray/pe/gcc/12.2.0/snos/bin/g++ python setup.py bdist_wheel"
####echo "BUILD_TEST=0 CUDAHOSTCXX=g++-14 CC=/usr/bin/gcc-14 CXX=/usr/bin/g++-14 python setup.py bdist_wheel"
#echo "CC=$(which cc) CXX=$(which CC) python setup.py bdist_wheel"
#CC=$(which cc) CXX=$(which CC) python setup.py bdist_wheel

# HARDCODE
#BUILD_TEST=0 CUDAHOSTCXX=g++-14 CC=/usr/bin/gcc-14 CXX=/usr/bin/g++-14 python setup.py bdist_wheel
export USE_MPI=1
BUILD_TEST=0 CUDAHOSTCXX=g++-14 CC=cc CXX=CC LDFLAGS="-L/opt/cray/pe/lib64 -Wl,-rpath,/opt/cray/pe/lib64 -lmpi_gtl_cuda ${LDFLAGS}" python setup.py bdist_wheel
PT_WHEEL=$(find dist/ -name "torch*.whl" -type f)
echo "copying pytorch wheel file $PT_WHEEL"
cp $PT_WHEEL $WHEELS_PATH/
cd $WHEELS_PATH
echo "pip installing $(basename $PT_WHEEL)"
pip install $(basename $PT_WHEEL)

cd $BASE_PATH
# KGF (2022-09-09):
MPICC="cc -shared -target-accel=nvidia80" pip install --force-reinstall --no-cache-dir --no-binary=mpi4py mpi4py

pip install ipython

echo "Cleaning up"
chmod -R u+w $DOWNLOAD_PATH/
# KGF: lot's of NFS errors lately with the following command
rm -rf $DOWNLOAD_PATH || true
# rm: cannot remove '/soft/applications/conda/2025-09-24/DOWNLOADS/.cache/bazel/_bazel_felker/8d63422c7e36f924c4f33033ca2fe451/server': Directory not empty
rm -rf $DOWNLOAD_PATH || true

conda list

chmod -R a-w $BASE_PATH/

set +e
