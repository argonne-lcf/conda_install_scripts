#!/bin/bash -l

# As of June 14 2021
# This script will install (from scratch) DeepHyper, TensorFlow, PyTorch, and Horovod on ThetaGPU
# 1 - Grab worker node interactively for 120 min (full-node queue)
# 2 - Run 'bash install_dh_hvd_tf_torch_v2.sh'
# 3 - script installs everything down in $PWD/deephyper/...
# 4 - wait for it to complete

# unset *_TAG variables to build latest master
DH_REPO_TAG="0.4.0"
DH_REPO_URL=https://github.com/deephyper/deephyper.git

#TF_REPO_TAG="e5a6d2331b11e0e5e4b63a0d7257333ac8b8262a" # requires NumPy 1.19.x
TF_REPO_TAG="v2.9.1"
PT_REPO_TAG="v1.12.0"
HOROVOD_REPO_TAG="v0.25.0" # v0.22.1 released on 2021-06-10 should be compatible with TF 2.6.x and 2.5.x
TF_REPO_URL=https://github.com/tensorflow/tensorflow.git
HOROVOD_REPO_URL=https://github.com/uber/horovod.git
PT_REPO_URL=https://github.com/pytorch/pytorch.git

# where to install relative to current path
# if [[ -z "$DH_REPO_TAG" ]]; then
#     DH_INSTALL_SUBDIR='2021-06-28/'
# else
#     DH_INSTALL_SUBDIR=deephyper/${DH_REPO_TAG}
# fi

DH_INSTALL_SUBDIR='2022-07-01/'

# MPI source on ThetaGPU
module list
module load openmpi/openmpi-4.1.4_ucx-1.12.1_gcc-9.4.0
module list
MPI=/lus/theta-fs0/software/thetagpu/openmpi/openmpi-4.1.4_ucx-1.12.1_gcc-9.4.0

# KGF(2022-07-01): should probably upgrade CUDA runtime and driver to 11.6
# CUDA path and version information
CUDA_VERSION_MAJOR=11
CUDA_VERSION_MINOR=4
CUDA_VERSION=$CUDA_VERSION_MAJOR.$CUDA_VERSION_MINOR
CUDA_BASE=/usr/local/cuda-$CUDA_VERSION

CUDA_DEPS_BASE=/lus/theta-fs0/software/thetagpu/cuda

CUDNN_VERSION_MAJOR=8
CUDNN_VERSION_MINOR=4
CUDNN_VERSION_EXTRA=1.50
CUDNN_VERSION=$CUDNN_VERSION_MAJOR.$CUDNN_VERSION_MINOR.$CUDNN_VERSION_EXTRA
#CUDNN_BASE=$CUDA_DEPS_BASE/cudnn-$CUDA_VERSION-linux-x64-v$CUDNN_VERSION
CUDNN_BASE=$CUDA_DEPS_BASE/cudnn-linux-x86_64-8.4.1.50_cuda11.6-archive

NCCL_VERSION_MAJOR=2
NCCL_VERSION_MINOR=12.12-1
NCCL_VERSION=$NCCL_VERSION_MAJOR.$NCCL_VERSION_MINOR
#NCCL_BASE=$CUDA_DEPS_BASE/nccl_$NCCL_VERSION+cuda${CUDA_VERSION}_x86_64
NCCL_BASE=$CUDA_DEPS_BASE/nccl_2.12.12-1+cuda11.6_x86_64
# KGF: no Extended Compatibility in NCCL --- use older NCCL version built with CUDA 11.0 until
# GPU device kernel driver upgraded from 11.0 ---> 11.4 in November 2021
#NCCL_BASE=$CUDA_DEPS_BASE/nccl_2.9.9-1+cuda11.0_x86_64

TENSORRT_VERSION_MAJOR=8
# 8.4.1.5 not yet supported in TF 2.9.1
# https://github.com/tensorflow/tensorflow/pull/56392
TENSORRT_VERSION_MINOR=2.5.1

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
export CUDA_TOOLKIT_PATH=$CUDA_BASE
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
export TF_CUDA_PATHS=$CUDA_BASE,$CUDNN_BASE,$NCCL_BASE,$TENSORRT_BASE
export GCC_HOST_COMPILER_PATH=$(which gcc)
export CC_OPT_FLAGS="-march=native -Wno-sign-compare"
export TF_SET_ANDROID_WORKSPACE=0

# get the folder where this script is living
if [ -n "$ZSH_EVAL_CONTEXT" ]; then
    THISDIR=$( cd "$( dirname "$0" )" && pwd -LP)
else  # bash, sh, etc.
    THISDIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd -LP)
fi
# KGF: why use -LP here? Aren't the flags more or less contradictory?
# THISDIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd -LP )

# set install path
DH_INSTALL_BASE_DIR=$THISDIR/$DH_INSTALL_SUBDIR
WHEEL_DIR=$DH_INSTALL_BASE_DIR/wheels

# confirm install path
echo On ThetaGPU $HOSTNAME
echo Installing module into $DH_INSTALL_BASE_DIR
#read -p "Are you sure? " -n 1 -r
#echo
#if [[ $REPLY =~ ^[Yy]$ ]]
#then
#    echo OK, you asked for it...
#else
#   exit -1
#fi


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


# set Conda installation folder and where downloaded content will stay
CONDA_PREFIX_PATH=$DH_INSTALL_BASE_DIR/mconda3
DOWNLOAD_PATH=$DH_INSTALL_BASE_DIR/DOWNLOADS

mkdir -p $CONDA_PREFIX_PATH
mkdir -p $DOWNLOAD_PATH

# Download and install conda for a base python installation
CONDAVER='py38_4.12.0'
# "latest" switched from Python 3.8.5 to 3.9.5 on 2021-07-21
# CONDAVER=latest
CONDA_DOWNLOAD_URL=https://repo.continuum.io/miniconda
CONDA_INSTALL_SH=Miniconda3-$CONDAVER-Linux-x86_64.sh
echo Downloading miniconda installer
wget $CONDA_DOWNLOAD_URL/$CONDA_INSTALL_SH -P $DOWNLOAD_PATH
chmod +x $DOWNLOAD_PATH/$CONDA_INSTALL_SH

echo Installing Miniconda
$DOWNLOAD_PATH/$CONDA_INSTALL_SH -b -p $CONDA_PREFIX_PATH -u

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

export LD_LIBRARY_PATH=$MPI/lib:$CUDA_BASE/lib64:$CUDNN_BASE/lib64:$NCCL_BASE/lib:$TENSORRT_BASE/lib
export PATH=$MPI/bin:\$PATH
EOF

# create custom pythonstart in local area to deal with python readlines error
cat > etc/pythonstart << EOF
# startup script for python to enable saving of interpreter history and
# enabling name completion

# import needed modules
import atexit
import os
#import readline
import rlcompleter

# where is history saved
historyPath = os.path.expanduser("~/.pyhistory")

# handler for saving history
def save_history(historyPath=historyPath):
    #import readline
    #try:
    #    readline.write_history_file(historyPath)
    #except:
    pass

# read history, if it exists
#if os.path.exists(historyPath):
#    readline.set_history_length(10000)
#    readline.read_history_file(historyPath)

# register saving handler
atexit.register(save_history)

# enable completion
#readline.parse_and_bind('tab: complete')

# cleanup
del os, atexit, rlcompleter, save_history, historyPath
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
env_prompt: "(${DH_INSTALL_SUBDIR}/{default_env}) "
pkgs_dirs:
   - \$HOME/.conda/pkgs
EOF

# move to base install directory
cd $DH_INSTALL_BASE_DIR

# setup conda environment
source $CONDA_PREFIX_PATH/setup.sh

# KGF: probably dont need a third (removed) network check--- proxy env vars inherited from either sourced setup.sh
# and/or first network check. Make sure "set+e" during above sourced setup.sh since the network check "wget" might
# return nonzero code if network is offline

echo CONDA BINARY: $(which conda)
echo CONDA VERSION: $(conda --version)
echo PYTHON VERSION: $(python --version)

cat > modulefile << EOF
#%Module2.0
## miniconda modulefile
##
proc ModulesHelp { } {
   puts stderr "This module will add Miniconda to your environment"
}

set _module_name  [module-info name]
set is_module_rm  [module-info mode remove]
set sys           [uname sysname]
set os            [uname release]
set HOME          $::env(HOME)

set CONDA_PREFIX                 $CONDA_PREFIX_PATH

setenv CONDA_PREFIX              \$CONDA_PREFIX
setenv PYTHONUSERBASE            \$HOME/.local/\${_module_name}
setenv ENV_NAME                  \$_module_name
setenv PYTHONSTARTUP             \$CONDA_PREFIX/etc/pythonstart

puts stdout "source \$CONDA_PREFIX/setup.sh"
module-whatis  "miniconda installation"
EOF

set -e

########
### Install TensorFlow
########

echo Conda install some dependencies

conda install -y cmake zip unzip ninja pyyaml mkl mkl-include setuptools cmake cffi typing_extensions future six requests dataclasses graphviz numba pymongo

# CUDA only: Add LAPACK support for the GPU if needed
# conda install -y -c pytorch magma-cuda${CUDA_VERSION_MAJOR}${CUDA_VERSION_MINOR}
# No magma-cuda114: https://anaconda.org/pytorch/repo
conda install -y -c pytorch magma-cuda116 #magma-cuda113

conda install -y -c conda-forge mamba
conda update -y pip

echo Clone TensorFlow
cd $DH_INSTALL_BASE_DIR
git clone $TF_REPO_URL
cd tensorflow

if [[ -z "$TF_REPO_TAG" ]]; then
    echo Checkout TensorFlow master
else
    echo Checkout TensorFlow tag $TF_REPO_TAG
    git checkout --recurse-submodules $TF_REPO_TAG
fi
BAZEL_VERSION=$(cat .bazelversion)
echo Found TensorFlow depends on Bazel version $BAZEL_VERSION

cd $DH_INSTALL_BASE_DIR
echo Download Bazel binaries
BAZEL_DOWNLOAD_URL=https://github.com/bazelbuild/bazel/releases/download/$BAZEL_VERSION
BAZEL_INSTALL_SH=bazel-$BAZEL_VERSION-installer-linux-x86_64.sh
BAZEL_INSTALL_PATH=$DH_INSTALL_BASE_DIR/bazel-$BAZEL_VERSION
wget $BAZEL_DOWNLOAD_URL/$BAZEL_INSTALL_SH -P $DOWNLOAD_PATH
chmod +x $DOWNLOAD_PATH/$BAZEL_INSTALL_SH
echo Intall Bazel in $BAZEL_INSTALL_PATH
bash $DOWNLOAD_PATH/$BAZEL_INSTALL_SH --prefix=$BAZEL_INSTALL_PATH
export PATH=$PATH:/$BAZEL_INSTALL_PATH/bin

cd $DH_INSTALL_BASE_DIR

echo Install TensorFlow Dependencies
# KGF: ignore $HOME/.local/lib/python3.8/site-packages pip user site-packages
export PYTHONNOUSERSITE=1
#pip install -U pip six 'numpy<1.19.0' wheel setuptools mock 'future>=0.17.1' 'gast==0.3.3' typing_extensions portpicker
# KGF: try relaxing the dependency verison requirements (esp NumPy, since PyTorch wants a later version?)
#pip install -U pip six 'numpy~=1.19.5' wheel setuptools mock future gast typing_extensions portpicker pydot
# KGF (2021-12-15): stop limiting NumPy for now. Unclear if problems with 1.20.3 and TF/Pytorch
pip install -U pip six numpy wheel setuptools mock future gast typing_extensions portpicker pydot packaging
pip install -U keras_applications --no-deps
pip install -U keras_preprocessing --no-deps

echo Configure TensorFlow

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
echo Bazel Build TensorFlow
# KGF: restrict Bazel to only see 32 cores of the dual socket 64-core (physical) AMD Epyc node (e.g. 256 logical cores).
# Else, Bazel will hit PID limit, even when set to 32,178 in /sys/fs/cgroup/pids/user.slice/user-XXXXX.slice/pids.max
# even if --jobs=500
HOME=$DOWNLOAD_PATH bazel build --jobs=500 --local_cpu_resources=32 --verbose_failures --config=cuda //tensorflow/tools/pip_package:build_pip_package

echo Build TensorFlow wheel
./bazel-bin/tensorflow/tools/pip_package/build_pip_package $WHEEL_DIR

echo Install TensorFlow
pip install $(find $WHEEL_DIR/ -name "tensorflow*.whl" -type f)
# KGF(2021-09-27): This installs Keras 2.6.0


########
### Install PyTorch
########

cd $DH_INSTALL_BASE_DIR
echo Clone PyTorch

git clone --recursive $PT_REPO_URL
cd pytorch
if [[ -z "$PT_REPO_TAG" ]]; then
    echo Checkout PyTorch master
else
    echo Checkout PyTorch tag $PT_REPO_TAG
    git checkout --recurse-submodules $PT_REPO_TAG
fi

echo "Install PyTorch"

# https://github.com/pytorch/pytorch/blob/master/setup.py
export CUDA_TOOLKIT_ROOT_DIR=$CUDA_BASE
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
cp $PT_WHEEL $WHEEL_DIR/
cd $WHEEL_DIR
echo "pip installing $(basename $PT_WHEEL)"
pip install $(basename $PT_WHEEL)


########
### Install Horovod
########

cd $DH_INSTALL_BASE_DIR

echo Clone Horovod

git clone --recursive $HOROVOD_REPO_URL
cd horovod

if [[ -z "$HOROVOD_REPO_TAG" ]]; then
    echo Checkout Horovod master
else
    echo Checkout Horovod tag $HOROVOD_REPO_TAG
    git checkout --recurse-submodules $HOROVOD_REPO_TAG
fi

echo "Build Horovod Wheel using MPI from $MPI and NCCL from ${NCCL_BASE}"
# https://horovod.readthedocs.io/en/stable/gpus_include.html
# If you installed NCCL 2 using the nccl-<version>.txz package, you should specify the path to NCCL 2 using the HOROVOD_NCCL_HOME environment variable.
# add the library path to LD_LIBRARY_PATH environment variable or register it in /etc/ld.so.conf.
export LD_LIBRARY_PATH=$MPI/lib:$NCCL_BASE/lib:$LD_LIBRARY_PATH
export PATH=$MPI/bin:$PATH

HOROVOD_CUDA_HOME=${CUDA_BASE} HOROVOD_NCCL_HOME=$NCCL_BASE HOROVOD_CMAKE=$(which cmake) HOROVOD_GPU_OPERATIONS=NCCL HOROVOD_WITH_TENSORFLOW=1 HOROVOD_WITH_PYTORCH=1 HOROVOD_WITHOUT_MXNET=1 python setup.py bdist_wheel
HVD_WHL=$(find dist/ -name "horovod*.whl" -type f)
cp $HVD_WHL $WHEEL_DIR/
HVD_WHEEL=$(find $WHEEL_DIR/ -name "horovod*.whl" -type f)
echo Install Horovod $HVD_WHEEL
pip install --force-reinstall $HVD_WHEEL

echo "Pip install TensorBoard profiler plugin"
pip install tensorboard_plugin_profile tensorflow_addons
echo "Pip install other packages"
pip install pandas h5py matplotlib scikit-learn scipy pytest
pip install sacred wandb # Denis requests, April 2022

echo "Adding module snooper so we can tell what modules people are using"
ln -s /lus/theta-fs0/software/datascience/PyModuleSnooper/sitecustomize.py $(python -c 'import site; print(site.getsitepackages()[0])')/sitecustomize.py

# DeepHyper stuff
export PATH=$MPI/bin:$PATH  # hvd optional feature will build mpi4py wheel

pip install 'tensorflow_probability==0.17.0'
# KGF: 0.17.0 (2022-06-06) tested against TF 2.9.1
# KGF: 0.14.0 (2021-09-15) only compatible with TF 2.6.0
# KGF: 0.13.0 (2021-06-18) only compatible with TF 2.5.0

if [[ -z "$DH_REPO_TAG" ]]; then
    echo Clone and checkout DeepHyper develop branch from git
    cd $DH_INSTALL_BASE_DIR
    git clone $DH_REPO_URL
    cd deephyper
    # KGF: use of GitFlow means that master branch might be too old for us:
    git checkout develop
    pip --version
    pip index versions deepspace
    pip install dh-scikit-optimize==0.9.0
    # Do not use editable pip installs
    # Uses deprecated egg format for symbolic link instead of wheels.
    # This causes permissions issues with read-only easy-install.pth
    pip install ".[analytics,hvd]"  # deepspace extra preseent in v0.3.3 but removed in develop branch
    cd ..
    cd $DH_INSTALL_BASE_DIR
else
    # hvd optional feature pinned to an old version in DH 0.2.5. Omit here
    echo Build DeepHyper tag $DH_REPO_TAG and Balsam from PyPI
    pip install "deephyper[nas,popt,autodeuq,analytics,hvd]==${DH_REPO_TAG}"  # otherwise, pulls 0.2.2 due to dependency conflicts?
fi


# random inconsistencies that pop up with the specific "pip installs" from earlier
pip install 'pytz>=2017.3' 'pillow>=6.2.0' 'django>=2.1.1'
# https://github.com/tensorflow/tensorflow/issues/46840#issuecomment-872946341
# https://github.com/pytorch/vision/issues/4146
# https://github.com/pytorch/vision/pull/4148
# https://github.com/pytorch/vision/issues/2632
pip install "pillow!=8.3.0,>=6.2.0"  # 8.3.1 seems to be fine with torchvision and dataloader
# KGF: torchvision will try to install its own .whl for PyTorch 1.9.0 even if 1.9.0a0+gitd69c22d is installed, e.g
pip install --no-deps torchvision
pip install --no-deps timm
pip install opencv-python-headless

pip install onnx
pip install onnxruntime-gpu  # onnxruntime is CPU-only. onnxruntime-gpu includes most CPU abilities
# https://github.com/microsoft/onnxruntime/issues/10685
# onnxruntime probably wont work on ThetaGPU single-gpu queue with CPU thread affinity
# https://github.com/microsoft/onnxruntime/issues/8313
pip install tf2onnx  # frontend for ONNX. tf->onnx
pip install onnx-tf  # backend (onnx->tf) and frontend (tf->onnx, deprecated) for ONNX
# https://github.com/onnx/onnx-tensorflow/issues/1010
# https://github.com/onnx/tensorflow-onnx/issues/1793
# https://github.com/onnx/onnx-tensorflow/issues/422
pip install transformers
pip install scikit-image
pip install torchinfo  # https://github.com/TylerYep/torchinfo successor to torchsummary (https://github.com/sksq96/pytorch-summary)
pip install cupy-cuda${CUDA_VERSION_MAJOR}${CUDA_VERSION_MINOR}
pip install pytorch-lightning

env MPICC=$MPI/bin/mpicc pip install mpi4py --no-cache-dir
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

echo Cleaning up
chmod -R u+w $DOWNLOAD_PATH/
rm -rf $DOWNLOAD_PATH

# KGF: see below
conda list

chmod -R a-w $DH_INSTALL_BASE_DIR/


set +e
# KGF: still need to apply manual postfix for the 4x following warnings that appear whenever "conda list" or other commands are run
# WARNING conda.gateways.disk.delete:unlink_or_rename_to_trash ... /lus/theta-fs0/software/thetagpu/conda/deephyper/0.2.5/mconda3/conda-meta/setuptools-52.0.0-py38h06a4308_0.json

# KGF: Do "chmod -R u+w ." in mconda3/conda-meta/, run "conda list", then "chmod -R a-w ."


# https://github.com/deephyper/deephyper/issues/110
# KGF: check that CONDA_DIR/mconda3/lib/python3.8/site-packages/easy-install.pth does not exist as an empty file
# rm it to prevent it from appearing in cloned conda environments (with read-only permissions), preventing users
# from instalilng editable pip installs in their own envs!
