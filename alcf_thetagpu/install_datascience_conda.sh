#!/bin/bash -l

# As of June 14 2021
# This script will install (from scratch) DeepHyper, TensorFlow, PyTorch, and Horovod on ThetaGPU
# 1 - Grab worker node interactively for 120 min (full-node queue)
# 2 - Run './<this script> /path/to/install/base/'
# 3 - script installs everything down in /path/to/install/base/
# 4 - wait for it to complete

# e.g. /lus/theta-fs0/software/thetagpu/conda/2023-01-11 on ThetaGPU, /soft/datascience/conda/2023-01-10 on Polaris
BASE_PATH=$1
DATE_PATH="$(basename $BASE_PATH)"

export PYTHONNOUSERSITE=1

# move primary conda packages directory/cache away from ~/.conda/pkgs (4.2 GB currently)
# hardlinks should be preserved even if these files are moved (not across filesystem boundaries)
export CONDA_PKGS_DIRS=/lus/theta-fs0/software/thetagpu/conda/pkgs

# unset *_TAG variables to build latest master/main branch (or "develop" in the case of DeepHyper)
#DH_REPO_TAG="0.4.2"
DH_REPO_URL=https://github.com/deephyper/deephyper.git

#TF_REPO_TAG="e5a6d2331b11e0e5e4b63a0d7257333ac8b8262a" # requires NumPy 1.19.x
TF_REPO_TAG="v2.11.0"
PT_REPO_TAG="v1.13.1"
HOROVOD_REPO_TAG="v0.27.0" # v0.22.1 released on 2021-06-10 should be compatible with TF 2.6.x and 2.5.x
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

# MPI source on ThetaGPU
module list
module load openmpi/openmpi-4.1.4_ucx-1.14.0_gcc-9.4.0_cuda-11.8
module list
MPI=/lus/theta-fs0/software/thetagpu/openmpi/openmpi-4.1.4_ucx-1.14.0_gcc-9.4.0_cuda-11.8

# KGF(2022-07-01): should probably upgrade CUDA runtime and driver to 11.6
# CUDA path and version information
CUDA_VERSION_MAJOR=11
CUDA_VERSION_MINOR=8
CUDA_VERSION_MINI=0
CUDA_VERSION=$CUDA_VERSION_MAJOR.$CUDA_VERSION_MINOR
CUDA_VERSION_FULL=$CUDA_VERSION.$CUDA_VERSION_MINI
#CUDA_TOOLKIT_BASE=/usr/local/cuda-$CUDA_VERSION

CUDA_DEPS_BASE=/lus/theta-fs0/software/thetagpu/cuda
CUDA_TOOLKIT_BASE=$CUDA_DEPS_BASE/cuda-$CUDA_VERSION_FULL

CUDNN_VERSION_MAJOR=8
CUDNN_VERSION_MINOR=6
CUDNN_VERSION_EXTRA=0.163
CUDNN_VERSION=$CUDNN_VERSION_MAJOR.$CUDNN_VERSION_MINOR.$CUDNN_VERSION_EXTRA
#CUDNN_BASE=$CUDA_DEPS_BASE/cudnn-$CUDA_VERSION-linux-x64-v$CUDNN_VERSION
CUDNN_BASE=$CUDA_DEPS_BASE/cudnn-$CUDA_VERSION_MAJOR-linux-x64-v$CUDNN_VERSION
# cudnn-linux-x86_64-8.6.0.163_cuda11-archive/
#CUDNN_BASE=$CUDA_DEPS_BASE/cudnn-linux-x86_64-8.4.1.50_cuda11.6-archive

NCCL_VERSION_MAJOR=2
NCCL_VERSION_MINOR=16.2-1
NCCL_VERSION=$NCCL_VERSION_MAJOR.$NCCL_VERSION_MINOR
NCCL_BASE=$CUDA_DEPS_BASE/nccl_$NCCL_VERSION+cuda${CUDA_VERSION}_x86_64
#NCCL_BASE=$CUDA_DEPS_BASE/nccl_2.12.12-1+cuda11.0_x86_64
# KGF: no Extended Compatibility in NCCL --- use older NCCL version built with CUDA 11.0 until
# GPU device kernel driver upgraded from 11.0 ---> 11.4 in November 2021
#NCCL_BASE=$CUDA_DEPS_BASE/nccl_2.9.9-1+cuda11.0_x86_64

TENSORRT_VERSION_MAJOR=8
# 8.4.1.5 not yet supported in TF 2.9.1
# https://github.com/tensorflow/tensorflow/pull/56392
TENSORRT_VERSION_MINOR=5.2.2

TENSORRT_VERSION=$TENSORRT_VERSION_MAJOR.$TENSORRT_VERSION_MINOR
#TENSORRT_BASE=$CUDA_DEPS_BASE/TensorRT-$TENSORRT_VERSION.Linux.x86_64-gnu.cuda-$CUDA_VERSION.cudnn$CUDNN_VERSION_MAJOR.$CUDNN_VERSION_MINOR
TENSORRT_BASE=$CUDA_DEPS_BASE/TensorRT-$TENSORRT_VERSION
# TensorFlow 2.7.0 does not yet support TRT 8.2.x as of 2021-11-30
# https://github.com/tensorflow/tensorflow/pull/52342
# support merged into master on 2021-11-11: https://github.com/tensorflow/tensorflow/pull/52932


# TensorFlow Config flags (for ./configure run)
export TF_CUDA_COMPUTE_CAPABILITIES=8.0
export TF_CUDA_VERSION=$CUDA_VERSION_MAJOR
export TF_CUDNN_VERSION=$CUDNN_VERSION_MAJOR
export TF_TENSORRT_VERSION=$TENSORRT_VERSION_MAJOR
export TF_NCCL_VERSION=$NCCL_VERSION_MAJOR
export CUDA_TOOLKIT_PATH=$CUDA_TOOLKIT_BASE
export CUDNN_INSTALL_PATH=$CUDNN_BASE
export NCCL_INSTALL_PATH=$NCCL_BASE
export TENSORRT_INSTALL_PATH=$TENSORRT_BASE
export TF_NEED_OPENCL_SYCL=0
export TF_NEED_COMPUTECPP=0
export TF_CUDA_CLANG=0
export TF_NEED_OPENCL=0
export TF_NEED_MPI=0
export TF_NEED_ROCM=0
export TF_NEED_CUDA=1
export TF_NEED_TENSORRT=1
export TF_CUDA_PATHS=$CUDA_TOOLKIT_BASE,$CUDNN_BASE,$NCCL_BASE,$TENSORRT_BASE
export GCC_HOST_COMPILER_PATH=$(which gcc)
export CC_OPT_FLAGS="-march=native -Wno-sign-compare"
export TF_SET_ANDROID_WORKSPACE=0

# Check for outside communication on ThetaGPU
# (be sure not to inherit these vars from dotfiles)
unset https_proxy
unset http_proxy

wget -q --spider -T 10 http://google.com
if [ $? -eq 0 ]; then
    echo "Network Online"
else
    # non-/interactive full-node job without --attrs=pubnet on ThetaGPU
    echo "Network Offline, setting proxy envs"
    export https_proxy=http://proxy.tmi.alcf.anl.gov:3128
    export http_proxy=http://proxy.tmi.alcf.anl.gov:3128
fi

# get the folder where this script is living
# if [ -n "$ZSH_EVAL_CONTEXT" ]; then
#     THISDIR=$( cd "$( dirname "$0" )" && pwd -LP)
# else  # bash, sh, etc.
#     THISDIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd -LP)
# fi
# KGF: why use -LP here? Aren't the flags more or less contradictory?
# THISDIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd -LP )

# set install path
####BASE_PATH=$THISDIR/$DH_INSTALL_SUBDIR


# confirm install path
echo "On ThetaGPU $HOSTNAME"
echo "Installing module into $BASE_PATH"

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

# Download and install conda for a base python installation
CONDAVER='py310_22.11.1-1'
# "latest" switched from Python 3.8.5 to 3.9.5 on 2021-07-21
# CONDAVER=latest
CONDA_DOWNLOAD_URL=https://repo.continuum.io/miniconda
CONDA_INSTALL_SH=Miniconda3-$CONDAVER-Linux-x86_64.sh
echo "Downloading miniconda installer"
wget $CONDA_DOWNLOAD_URL/$CONDA_INSTALL_SH -P $DOWNLOAD_PATH
chmod +x $DOWNLOAD_PATH/$CONDA_INSTALL_SH

echo "Installing Miniconda"
echo "bash $DOWNLOAD_PATH/$CONDA_INSTALL_SH -b -p $CONDA_PREFIX_PATH -u"
bash $DOWNLOAD_PATH/$CONDA_INSTALL_SH -b -p $CONDA_PREFIX_PATH -u

cd $CONDA_PREFIX_PATH

# create a setup file
cat > setup.sh << EOF
preferred_shell=\$(basename \$SHELL)

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
   export https_proxy=http://proxy.tmi.alcf.anl.gov:3128
   export http_proxy=http://proxy.tmi.alcf.anl.gov:3128
fi

export CUDA_TOOLKIT_BASE=$CUDA_TOOLKIT_BASE
export CUDNN_BASE=$CUDNN_BASE
export NCCL_BASE=$NCCL_BASE
export TENSORRT_BASE=$TENSORRT_BASE

# original thetagpu script adds MPI, not in Polaris script
export LD_LIBRARY_PATH=$MPI/lib:\$CUDA_TOOLKIT_BASE/lib64:\$CUDNN_BASE/lib:\$NCCL_BASE/lib:\$TENSORRT_BASE/lib:\$LD_LIBRARY_PATH:
export PATH=$MPI/lib:\$CUDA_TOOLKIT_BASE/bin:\$PATH

EOF

# KGF: $CONDA_ENV (e.g. conda/2021-11-30) is not an official conda var; set by us in modulefile
# $CONDA_DEFAULT_ENV (short name of current env) and $CONDA_PREFIX (full path) are official,
# but barely documented. powerlevel10k wont parse env variables when outputting the prompt,
# so best not to leave \$CONDA_ENV unparsed in env_prompt
# https://github.com/romkatv/powerlevel10k/issues/762#issuecomment-633389123
# # env_prompt (str)
# #   Template for prompt modification based on the active environment.
# #   Currently supported template variables are '{prefix}', '{name}', and
# #   '{default_env}'. '{prefix}' is the absolute path to the active
# #   environment. '{name}' is the basename of the active environment
# #   prefix. '{default_env}' holds the value of '{name}' if the active
# #   environment is a conda named environment ('-n' flag), or otherwise
# #   holds the value of '{prefix}'. Templating uses python's str.format()
# #   method.
cat > .condarc << EOF
channels:
   - defaults
   - pytorch
   - conda-forge
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

# KGF: probably dont need a third (removed) network check--- proxy env vars inherited from either sourced setup.sh
# and/or first network check. Make sure "set+e" during above sourced setup.sh since the network check "wget" might
# return nonzero code if network is offline

echo "CONDA BINARY: $(which conda)"
echo "CONDA VERSION: $(conda --version)"
echo "PYTHON VERSION: $(python --version)"

set -e

################################################
### Install TensorFlow
################################################

echo "Conda install some dependencies"

# polaris has /usr/bin/git-lfs, thetagpu does not
conda install -y -c defaults -c pytorch -c conda-forge cmake zip unzip astunparse ninja setuptools future six requests dataclasses graphviz numba numpy pymongo conda-build pip libaio mkl mkl-include onednn mkl-dnn git-lfs magma-cuda${CUDA_VERSION_MAJOR}${CUDA_VERSION_MINOR}

# note, numba pulls in numpy here too
#####conda install -y -c defaults -c conda-forge cmake zip unzip astunparse ninja setuptools future six requests dataclasses graphviz numba numpy pymongo conda-build pip libaio git-lfs
#####conda install -y -c defaults -c conda-forge mkl mkl-include onednn mkl-dnn
##### conda install -y cffi typing_extensions pyyaml # KGF: stopped using

# KGF: note, ordering of the above "defaults" channel install relative to "conda install -y -c conda-forge mamba; conda install -y pip"
# (used to leave the pip re-install on a separate line) may affect what version of numpy you end up with
# E.g. Jan 2023, Polaris ordering (defaults, then mamba then pip) got numpy 1.23.5 and ThetaGPU (mamba, pip, then defaults) got numpy 1.21.5

# CUDA only: Add LAPACK support for the GPU if needed
#####conda install -y -c defaults -c pytorch -c conda-forge magma-cuda${CUDA_VERSION_MAJOR}${CUDA_VERSION_MINOR}
# No magma-cuda114: https://anaconda.org/pytorch/repo ; but magma-cuda116 #magma-cuda113

#####conda install -y -c defaults -c conda-forge mamba

echo "Clone TensorFlow"
cd $BASE_PATH
git clone $TF_REPO_URL
cd tensorflow

if [[ -z "$TF_REPO_TAG" ]]; then
    echo "Checkout TensorFlow master"
else
    echo "Checkout TensorFlow tag $TF_REPO_TAG"
    git checkout --recurse-submodules $TF_REPO_TAG
fi
BAZEL_VERSION=$(cat .bazelversion)
echo "Found TensorFlow depends on Bazel version $BAZEL_VERSION"

cd $BASE_PATH
echo "Download Bazel binaries"
BAZEL_DOWNLOAD_URL=https://github.com/bazelbuild/bazel/releases/download/$BAZEL_VERSION
BAZEL_INSTALL_SH=bazel-$BAZEL_VERSION-installer-linux-x86_64.sh
BAZEL_INSTALL_PATH=$BASE_PATH/bazel-$BAZEL_VERSION
wget $BAZEL_DOWNLOAD_URL/$BAZEL_INSTALL_SH -P $DOWNLOAD_PATH
chmod +x $DOWNLOAD_PATH/$BAZEL_INSTALL_SH
echo "Intall Bazel in $BAZEL_INSTALL_PATH"
bash $DOWNLOAD_PATH/$BAZEL_INSTALL_SH --prefix=$BAZEL_INSTALL_PATH
export PATH=$PATH:/$BAZEL_INSTALL_PATH/bin

cd $BASE_PATH

echo "Install TensorFlow Dependencies"
# KGF: ignore $HOME/.local/lib/python3.8/site-packages pip user site-packages
#pip install -U pip six 'numpy<1.19.0' wheel setuptools mock 'future>=0.17.1' 'gast==0.3.3' typing_extensions portpicker
# KGF: try relaxing the dependency verison requirements (esp NumPy, since PyTorch wants a later version?)
#pip install -U pip six 'numpy~=1.19.5' wheel setuptools mock future gast typing_extensions portpicker pydot
# KGF (2021-12-15): stop limiting NumPy for now. Unclear if problems with 1.20.3 and TF/Pytorch
pip install -U numpy numba
# the above line can be very important or very bad, to get have pip control the numpy dependency chain right before TF build
# Start including numba here too in order to ensure mutual compat; numba 0.56.4 req numpy <1.24.0, e.g.
# Check https://github.com/numpy/numpy/blob/main/numpy/core/setup_common.py
# for C_API_VERSION, and track everytime numpy is reinstalled in the build log

# consistent with polaris:
pip install -U pip wheel mock gast portpicker pydot packaging pyyaml
# pip install -U pip pyyaml six wheel setuptools mock future gast typing_extensions portpicker pydot packaging
pip install -U keras_applications --no-deps
pip install -U keras_preprocessing --no-deps

echo "Configure TensorFlow"

# KGF: avoid problem when building post 2.5.0 master (762790710496e04801ec17dd7f012fd9aa37a1df)
# ERROR: /lus/theta-fs0/software/thetagpu/conda/deephyper/latest/tensorflow/tensorflow/core/kernels/mlir_generated/BUILD:910:23: compile tensorflow/core/kernels/mlir_generated/bitwise_xor_gpu_i32_i32_kernel_generator_kernel.o failed (Exit 1): tf_to_kernel failed: error executing command bazel-out/k8-opt/bin/tensorflow/compiler/mlir/tools/kernel_gen/tf_to_kernel '--tile_sizes=1024' '--max-supported-rank=5' '--arch=compute_80' ... (remaining 4 argument(s) skipped)
# 2021-06-21 21:49:28.967640: I tensorflow/compiler/mlir/tensorflow/utils/dump_mlir_util.cc:210] disabling MLIR crash reproducer, set env var `MLIR_CRASH_REPRODUCER_DIRECTORY` to enable.
# 2021-06-21 21:49:29.052480: W tensorflow/compiler/mlir/tools/kernel_gen/kernel_creator.cc:385] There should be exactly one GPU Module, but got 8. Currently we leak memory if there is more than one module, see https://bugs.llvm.org/show_bug.cgi?id=48385
export CUDA_VISIBLE_DEVICES=0


cd tensorflow
export PYTHON_BIN_PATH=$(which python)
export PYTHON_LIB_PATH=$(python -c 'import site; print(site.getsitepackages()[0])')
# Auto-Configuration Warning: 'TMP' environment variable is not set, using 'C:\Windows\Temp' as default
export TMP=/tmp

./configure
echo "Bazel Build TensorFlow"
# KGF: restrict Bazel to only see 32 cores of the dual socket 64-core (physical) AMD Epyc node (e.g. 256 logical cores).
# Else, Bazel will hit PID limit, even when set to 32,178 in /sys/fs/cgroup/pids/user.slice/user-XXXXX.slice/pids.max
# even if --jobs=500
HOME=$DOWNLOAD_PATH bazel build --jobs=500 --local_cpu_resources=32 --verbose_failures --config=cuda //tensorflow/tools/pip_package:build_pip_package

echo "Build TensorFlow wheel"
./bazel-bin/tensorflow/tools/pip_package/build_pip_package $WHEELS_PATH

echo "Install TensorFlow"
pip install $(find $WHEELS_PATH/ -name "tensorflow*.whl" -type f)
# KGF(2021-09-27): This installs Keras 2.6.0


#################################################
### Install PyTorch
#################################################

cd $BASE_PATH
echo "Clone PyTorch"

git clone --recursive $PT_REPO_URL
cd pytorch
if [[ -z "$PT_REPO_TAG" ]]; then
    echo "Checkout PyTorch master"
else
    echo "Checkout PyTorch tag $PT_REPO_TAG"
    git checkout --recurse-submodules $PT_REPO_TAG
    git submodule sync
    git submodule update --init --recursive
fi

echo "Install PyTorch"

# https://github.com/pytorch/pytorch/blob/master/setup.py
export CUDA_TOOLKIT_ROOT_DIR=$CUDA_TOOLKIT_BASE
export NCCL_ROOT_DIR=$NCCL_BASE
export CUDNN_ROOT=$CUDNN_BASE
export USE_TENSORRT=ON
export TENSORRT_ROOT=$TENSORRT_BASE
export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
#export TENSORRT_LIBRARY=$TENSORRT_BASE/lib/libmyelin.so
export TENSORRT_LIBRARY=$TENSORRT_BASE/lib
export TENSORRT_LIBRARY_INFER=$TENSORRT_BASE/lib/libnvinfer.so
export TENSORRT_LIBRARY_INFER_PLUGIN=$TENSORRT_BASE/lib/libnvinfer_plugin.so
export TENSORRT_INCLUDE_DIR=$TENSORRT_BASE/include
# KGF:
# pytorch/cmake/public/cuda.cmake
  # 116   find_path(TENSORRT_INCLUDE_DIR NvInfer.h
  # 117     HINTS ${TENSORRT_ROOT} ${CUDA_TOOLKIT_ROOT_DIR}
  # 118     PATH_SUFFIXES include)
  # 119   find_library(TENSORRT_LIBRARY nvinfer
  # 120     HINTS ${TENSORRT_ROOT} ${CUDA_TOOLKIT_ROOT_DIR}
  # 121     PATH_SUFFIXES lib lib64 lib/x64)
  # 122   find_package_handle_standard_args(
  # 123     TENSORRT DEFAULT_MSG TENSORRT_INCLUDE_DIR TENSORRT_LIBRARY)
  # 124   if(TENSORRT_FOUND)
# ---> Could NOT find TENSORRT (missing: TENSORRT_INCLUDE_DIR TENSORRT_LIBRARY)
# USE_TENSORRT: OFF in the Summary
# but then later: cmake ... -DUSE_TENSORRT=ON

# BUILD_CAFFE2=0
#CFLAGS="-Wextra -DOMPI_SKIP_MPICXX"
CFLAGS="-DOMPI_SKIP_MPICXX" CC=$MPI/bin/mpicc CXX=$MPI/bin/mpic++ python setup.py bdist_wheel
# KGF: might need to unload the default openmpi/openmpi-4.0.5 module (didn't work)
# and/or modify LD_LIBRARY_PATH
# https://github.com/pytorch/pytorch#adjust-build-options-optional
# https://discuss.pytorch.org/t/mpi-pytorch-backend/18988
# https://github.com/pytorch/pytorch/blob/master/cmake/Summary.cmake
PT_WHEEL=$(find dist/ -name "torch*.whl" -type f)
echo "Copying pytorch wheel file $PT_WHEEL"
cp $PT_WHEEL $WHEELS_PATH/
cd $WHEELS_PATH
echo "pip installing $(basename $PT_WHEEL)"
pip install $(basename $PT_WHEEL)


#################################################
### Install Horovod
#################################################

cd $BASE_PATH

echo "Clone Horovod"

git clone --recursive $HOROVOD_REPO_URL
cd horovod

if [[ -z "$HOROVOD_REPO_TAG" ]]; then
    echo "Checkout Horovod master"
else
    echo "Checkout Horovod tag $HOROVOD_REPO_TAG"
    git checkout --recurse-submodules $HOROVOD_REPO_TAG
fi

echo "Build Horovod Wheel using MPI from $MPI and NCCL from ${NCCL_BASE}"
# https://horovod.readthedocs.io/en/stable/gpus_include.html
# If you installed NCCL 2 using the nccl-<version>.txz package, you should specify the path to NCCL 2 using the HOROVOD_NCCL_HOME environment variable.
# add the library path to LD_LIBRARY_PATH environment variable or register it in /etc/ld.so.conf.
export LD_LIBRARY_PATH=$MPI/lib:$NCCL_BASE/lib:$LD_LIBRARY_PATH
export PATH=$MPI/bin:$PATH

HOROVOD_CUDA_HOME=${CUDA_TOOLKIT_BASE} HOROVOD_NCCL_HOME=$NCCL_BASE HOROVOD_CMAKE=$(which cmake) HOROVOD_GPU_OPERATIONS=NCCL HOROVOD_WITH_TENSORFLOW=1 HOROVOD_WITH_PYTORCH=1 HOROVOD_WITHOUT_MXNET=1 python setup.py bdist_wheel
HVD_WHL=$(find dist/ -name "horovod*.whl" -type f)
cp $HVD_WHL $WHEELS_PATH/
HVD_WHEEL=$(find $WHEELS_PATH/ -name "horovod*.whl" -type f)
echo "Install Horovod $HVD_WHEEL"
pip install --force-reinstall --no-cache-dir $HVD_WHEEL

echo "Pip install TensorBoard profiler plugin"
pip install tensorboard_plugin_profile tensorflow_addons tensorflow-datasets
echo "Pip install other packages"
pip install pandas h5py matplotlib scikit-learn scipy pytest
pip install sacred wandb # Denis requests, April 2022

echo "Adding module snooper so we can tell what modules people are using"
# KGF: TODO, modify this path at the top of the script somehow; pick correct sitecustomize_polaris.py, etc.
# wont error out if first path does not exist; will just make a broken symbolic link
ln -s /lus/theta-fs0/software/datascience/PyModuleSnooper/sitecustomize.py $(python -c 'import site; print(site.getsitepackages()[0])')/sitecustomize.py

# DeepHyper stuff
export PATH=$MPI/bin:$PATH  # hvd optional feature will build mpi4py wheel

pip install 'tensorflow_probability==0.19.0'
# KGF: 0.17.0 (2022-06-06) tested against TF 2.9.1
# KGF: 0.14.0 (2021-09-15) only compatible with TF 2.6.0
# KGF: 0.13.0 (2021-06-18) only compatible with TF 2.5.0

if [[ -z "$DH_REPO_TAG" ]]; then
    echo "Clone and checkout DeepHyper develop branch from git"
    cd $BASE_PATH
    git clone $DH_REPO_URL
    cd deephyper
    # KGF: use of GitFlow means that master branch might be too old for us:
    git checkout develop
    #     pip --version
    #     pip index versions deepspace
    #     pip install dh-scikit-optimize==0.9.0

    # Do not use editable pip installs
    # Uses deprecated egg format for symbolic link instead of wheels.
    # This causes permissions issues with read-only easy-install.pth
    # -----------------
    pip install ".[analytics,hvd,nas,popt,autodeuq]"
    # Adding "sdv" optional requirement on Polaris with Python 3.8 force re-installed:
    # numpy-1.22.4, torch-1.13.1, which requires nvidia-cuda-nvrtc-cu11 + many other deps
    # No problem on ThetaGPU. Switching to Python 3.10 apparently avoids everything
    # TODO: if problems start again, test installing each of the sdv deps one-by-one (esp. ctgan)
    pip install ".[analytics,hvd,nas,popt,autodeuq,sdv]"
    cd ..
    cd $BASE_PATH
else
    # hvd optional feature pinned to an old version in DH 0.2.5. Omit here
    echo "Build DeepHyper tag $DH_REPO_TAG and Balsam from PyPI"
    #pip install "deephyper[analytics,hvd,nas,popt,autodeuq]==${DH_REPO_TAG}"
    pip install "deephyper[analytics,hvd,nas,popt,autodeuq,sdv]==${DH_REPO_TAG}"  # otherwise, pulls 0.2.2 due to dependency conflicts?
fi

pip install 'libensemble'


# PyTorch Geometric--- Hardcoding PyTorch 1.13.0, CUDA 11.7  even though installing Pytorch 1.13.1, CUDA 11.8 above
pip install pyg-lib torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-1.13.0+cu117.html
#pip install pyg-lib torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-1.13.0+cu${CUDA_VERSION_MAJOR}${CUDA_VERSION_MINOR}.html
pip install torch-geometric

# random inconsistencies that pop up with the specific "pip installs" from earlier
pip install 'pytz>=2017.3' 'pillow>=6.2.0' 'django>=2.1.1'
# https://github.com/tensorflow/tensorflow/issues/46840#issuecomment-872946341
# https://github.com/pytorch/vision/issues/4146
# https://github.com/pytorch/vision/pull/4148
# https://github.com/pytorch/vision/issues/2632
pip install "pillow!=8.3.0,>=6.2.0"  # 8.3.1 seems to be fine with torchvision and dataloader
# KGF: torchvision will try to install its own .whl for PyTorch 1.9.0 even if 1.9.0a0+gitd69c22d is installed, e.g
#pip install --no-deps torchvision
# KGF Polaris: need exact CUDA minor version match, and torch 1.12.0 needs vision 1.13.0
# https://github.com/pytorch/vision#installation
# PyTorch: you could check the linked CUDA version via print(torch.version.cuda)

#pip install --no-dependencies torchvision==0.13.0+cu115 --extra-index-url https://download.pytorch.org/whl/
# No 0.13.0+cu115 prebuilt binary, only cu116 and older ones. Must build from source
# https://download.pytorch.org/whl/torch_stable.html

cd $BASE_PATH
echo "Install PyTorch Vision from source"
git clone https://github.com/pytorch/vision.git
cd vision
git checkout v0.14.1
# KGF: this falls back to building a deprecated .egg format with easy_install, which puts an entry in
# mconda3/lib/python3.8/site-packages/easy-install.pth, causing read-only premissions problems in cloned
# environments.
###python setup.py install
# "We don't officially support building from source using pip, but if you do, you'll need to use the --no-build-isolation flag."

# KGF: build our own wheel, like in PyTorch and TF builds:
python setup.py bdist_wheel
VISION_WHEEL=$(find dist/ -name "torchvision*.whl" -type f)
cp $VISION_WHEEL $WHEELS_PATH/
cd $WHEELS_PATH
echo "pip installing $(basename $VISION_WHEEL)"
# KGF: unlike "python setup.py install", still tries to install PyTorch again by default, despite being a local wheel
pip install --force-reinstall --no-deps $(basename $VISION_WHEEL)

cd $BASE_PATH

pip install --no-deps timm
pip install opencv-python-headless

# onnx 1.13.0 pushes protobuf to >3.20.2 and "tensorflow 2.11.0 requires protobuf<3.20,>=3.9.2, but you have protobuf 3.20.3 which is incompatible."
#  onnx runtime 1.13.1 pushes numpy>=1.21.6, which installs 1.24.x for some reason, breaking <1.22 compat with numba
pip install 'onnx==1.12.0' 'onnxruntime-gpu==1.12.1'
# onnxruntime is CPU-only. onnxruntime-gpu includes most CPU abilities
# https://github.com/microsoft/onnxruntime/issues/10685
# onnxruntime probably wont work on ThetaGPU single-gpu queue with CPU thread affinity
# https://github.com/microsoft/onnxruntime/issues/8313
pip install tf2onnx  # frontend for ONNX. tf->onnx
pip install onnx-tf  # backend (onnx->tf) and frontend (tf->onnx, deprecated) for ONNX
# https://github.com/onnx/onnx-tensorflow/issues/1010
# https://github.com/onnx/tensorflow-onnx/issues/1793
# https://github.com/onnx/onnx-tensorflow/issues/422
pip install huggingface-hub
pip install transformers evaluate datasets accelerate
pip install scikit-image
pip install line_profiler
pip install torch-tb-profiler
pip install torchinfo  # https://github.com/TylerYep/torchinfo successor to torchsummary (https://github.com/sksq96/pytorch-summary)
pip install cupy-cuda${CUDA_VERSION_MAJOR}x
pip install pycuda
pip install pytorch-lightning
pip install ml-collections
pip install gpytorch xgboost multiprocess py4j
pip install hydra-core hydra_colorlog accelerate arviz pyright celerite seaborn xarray bokeh matplotx aim torchviz rich parse

# PyPI binary wheels 1.1.1, 1.0.0 might only work with CPython 3.6-3.9, not 3.10
#pip install "triton==1.0.0"
#pip install 'triton==2.0.0.dev20221202' || true

# But, DeepSpeed sparse support only supports triton v1.0
cd $BASE_PATH
echo "Install triton v1.0 from source"
git clone https://github.com/openai/triton.git
cd triton/python
git checkout v1.0
pip install .
#pip install deepspeed

cd $BASE_PATH
echo "Install DeepSpeed from source"
git clone https://github.com/microsoft/DeepSpeed.git
cd DeepSpeed
export CFLAGS="-I${CONDA_PREFIX}/include/"
#export LDFLAGS="-L${CONDA_PREFIX}/lib/"
export LDFLAGS="-L${CONDA_PREFIX}/lib/ -Wl,--enable-new-dtags,-rpath,${CONDA_PREFIX}/lib"
#DS_BUILD_OPS=1 DS_BUILD_AIO=1 DS_BUILD_UTILS=1 bash install.sh --verbose
DS_BUILD_OPS=1 DS_BUILD_AIO=1 DS_BUILD_UTILS=1 pip install .
# if no rpath, add this to Lmod modulefile:
#export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${CONDA_PREFIX}/lib"
# Suboptimal, since we really only want "libaio.so" from that directory to run DeepSpeed.
# e.g. if you put "export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}", it overrides many system libraries
# breaking "module list", "emacs", etc.

cd $BASE_PATH

pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
#pip install "jax[cuda11_cudnn82]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install pymongo optax flax
pip install "numpyro[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# note, polaris script installs mpi4py after Horovod, before DeepHyper, PyTorch vision, ...
env MPICC=$MPI/bin/mpicc pip install mpi4py --force-reinstall --no-cache-dir --no-binary=mpi4py
# conda install -c conda-forge cupy cudnn cutensor nccl
# https://github.com/cupy/cupy/issues/4850
## https://docs.cupy.dev/en/stable/install.html?highlight=cutensor#additional-cuda-libraries
# KGF: installed CuPy 10.0.0, no cuTENSOR, cuSPARSELt installed

# Reason: ImportError (libcutensor.so.1: cannot open shared object file: No such file or directory)
# python -m cupyx.tools.install_library --library cutensor --cuda 11.4

# import cupy.cuda.cudnn
# import cupy.cuda.nccl
# cupy.cuda.cudnn.getVersion()
#       8300 (does NOT match version 8.2.4.15 in /lus/theta-fs0/software/thetagpu/cuda/ that conda/2021-11-30 was built with)
# cupy.cuda.nccl.get_version()
#       21104 (matches version in /lus/theta-fs0/software/thetagpu/cuda/ ...)

# https://docs.cupy.dev/en/stable/upgrade.html?highlight=cutensor#compatibility-matrix
# https://docs.cupy.dev/en/stable/reference/environment.html?highlight=cutensor#envvar-CUTENSOR_PATH

pip install cython
git clone https://github.com/mpi4jax/mpi4jax.git
cd mpi4jax
CUDA_ROOT=$CUDA_TOOLKIT_BASE pip install --no-build-isolation --no-cache-dir --no-binary=mpi4jax -v .
cd $BASE_PATH

# ------------------------------------------------
# KGF: unreleased tf sometimes pulls in keras-nightly, which confuses Horovod with the standalone Keras (usually installed as a dependency of DeepHyper). But it seems necessary in order to run the resulting Horovod installation
####pip uninstall -y 'keras' || true
# KGF: the above line might not work. Double check with "horovodrun --check-build". Confirmed working version of keras-nightly as of 2021-07-14
#####pip install 'keras-nightly~=2.6.0.dev2021052700' || true

# KGF(2021-09-27): Confusingly, these commands worked for a fresh install of TF 2.6.0, resulting in only keras-nightly, not keras, installed in Conda. However, when I went to modify the existing conda environment to 'pip install -e ".[analytics,deepspace,hvd]"' a newer version of DeepHyper, it reinstalled Keras 2.6.0, which I then manually uninstalled.

# This broke "horovodrun --check-build" TensorFlow integration, and you could no longer even import tensorflow.

# Uninstalling "keras-nightly" and reinstalling "Keras" seems to fix this, even though it is the opposite setup from the original (working) install script. Seems to be a different behavior depending on whether or not the TensorFlow build is from a tagged release vs. unstable master. E.g. conda/2021-06-26 (tagged version) installed keras 2.4.3, conda/2021-06-28 installed keras-nightly 2.6.0.dev2021062500

# Where does TensorFlow define a Keras dependency when you build a wheel from source??
# ANSWER: https://github.com/tensorflow/tensorflow/commit/e457b3604ac31e7e0e38eaae8622509302f8c7d6#diff-f526feeafa1000c4773410bdc5417c4022cb2c7b686ae658b629beb541ae9112
# They were temporarily using keras-nightly for the dep; switched away from that on 2021-08-09.

echo "Cleaning up"
chmod -R u+w $DOWNLOAD_PATH/
rm -rf $DOWNLOAD_PATH

# KGF: see below
conda list

chmod -R a-w $BASE_PATH/


set +e
# KGF: still need to apply manual postfix for the 4x following warnings that appear whenever "conda list" or other commands are run
# WARNING conda.gateways.disk.delete:unlink_or_rename_to_trash ... /lus/theta-fs0/software/thetagpu/conda/deephyper/0.2.5/mconda3/conda-meta/setuptools-52.0.0-py38h06a4308_0.json

# KGF: Do "chmod -R u+w ." in mconda3/conda-meta/, run "conda list", then "chmod -R a-w ."


# https://github.com/deephyper/deephyper/issues/110
# KGF: check that CONDA_DIR/mconda3/lib/python3.8/site-packages/easy-install.pth does not exist as an empty file
# rm it to prevent it from appearing in cloned conda environments (with read-only permissions), preventing users
# from instalilng editable pip installs in their own envs!
