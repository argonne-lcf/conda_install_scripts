#!/bin/bash -l

# As of May 2022
# This script will install TensorFlow, PyTorch, and Horovod on Polaris, all from source
# 1 - Login to Polaris login-node
# 2 - Run './<this script> /path/to/install/base/'
# 3 - script installs everything down in /path/to/install/base/
# 4 - wait for it to complete

# KGF: check HARDCODE points for lines that potentially require manual edits to pinned package versions

BASE_PATH=/soft/datascience/conda/2023-09-29/
DATE_PATH="$(basename $BASE_PATH)"
DOWNLOAD_PATH=$BASE_PATH/DOWNLOADS

export PYTHONNOUSERSITE=1
# KGF: PBS used to mess with user umask, changing it to 0077 on compute node
# dirs that were (2555/dr-xr-sr-x) on ThetaGPU became (2500/dr-x--S---)
umask 0022

# move primary conda packages directory/cache away from ~/.conda/pkgs (4.2 GB currently)
# hardlinks should be preserved even if these files are moved (not across filesystem boundaries)
export CONDA_PKGS_DIRS="/soft/datascience/conda/pkgs"

export CONDA_PREFIX="/soft/datascience/conda/2023-09-29/mconda3"
export CONDA_PREFIX_PATH=$CONDA_PREFIX

module load gcc-mixed # get 11.2.0 (2021) instead of /usr/bin/gcc 7.5 (2019)
module load craype-accel-nvidia80  # wont load for PrgEnv-gnu; see HPE Case 5367752190
export MPICH_GPU_SUPPORT_ENABLED=1

module unload gcc-mixed
module load PrgEnv-gnu

# setup conda environment
source $CONDA_PREFIX_PATH/setup.sh
echo "after sourcing conda"
set -e

cd $BASE_PATH
# echo "Install CUTLASS from source"
# #git clone https://github.com/NVIDIA/cutlass
# cd cutlass
export CUTLASS_PATH="${BASE_PATH}/cutlass"
export CUDA_TOOLKIT_BASE="/soft/compilers/cudatoolkit/cuda-11.8.0"
export CUDA_HOME=${CUDA_TOOLKIT_BASE}


# https://github.com/NVIDIA/cutlass/blob/main/media/docs/quickstart.md
# mkdir build || true
# cd build
# strangely, https://github.com/NVIDIA/cutlass/blob/main/cuDNN.cmake doesnt respect $CUDNN_INCLUDE_DIR
export CUDNN_PATH="/soft/libraries/cudnn/cudnn-11-linux-x64-v8.6.0.163/"
export CUDNN_BASE=$CUDNN_PATH
export CUDNN_INCLUDE_DIR=$CUDNN_BASE/include
export CPATH="$CPATH:$CUDNN_INCLUDE_DIR"

export CUDA_INSTALL_PATH=${CUDA_HOME}
export CUDACXX=${CUDA_INSTALL_PATH}/bin/nvcc
# echo "About to run CMake for CUTLASS python = $(which python)"
# conda info
# cmake .. -DCUTLASS_NVCC_ARCHS=80 -DCUTLASS_ENABLE_CUBLAS=ON -DCUTLASS_ENABLE_CUDNN=ON
# KGF: spurious errors with above CUTLASS cmake command in script (never encountered in interactive job)
# CMake Error at tools/library/CMakeLists.txt:285 (message):
#   Error generating library instances.  See
#   /soft/datascience/conda/2023-09-28/cutlass/build/tools/library/library_instance_generation.log
# ... (in that file:)
#

# KGF issue https://github.com/NVIDIA/cutlass/issues/1118
#echo "apply patch for CUTLASS #include <limits> in platform.h"
#cd ..
#git apply ~/cutlass_limits.patch
#cd build
#make cutlass_profiler -j32  # fails at 98% cudnn_helpers.cpp.o without patch
#make test_unit -j32  # this passes

# https://github.com/NVIDIA/cutlass/blob/main/python/README.md
# export CUDA_INSTALL_PATH=${CUDA_HOME}
# if unset, it defaults to:
#        which nvcc | awk -F'/bin/nvcc' '{print $1}'
##python setup.py develop --user
# KGF: both "develop" and "--user" installs are problematic for our shared Anaconda envs
# "We plan to add support for installing via python setup.py install in a future release."

# -----------------

# DeepSpeed 0.10.3 (2023-09-11) notes:
# 1) https://github.com/microsoft/DeepSpeed/issues/3491
# DS_BUILD_SPARSE_ATTN=0 because this feature requires triton 1.x (old), while Stable Diffusion requires triton 2.x
# 2) do we really care about JIT vs. precompiled features?
# 3) DS_BUILD_UTILS=1 ---> what does it include? does it imply DS_BUILD_EVOFORMER_ATTN=1? No; see below
# 4) -j32 might be too much? overall, "pip install deepspeed" appears more stable than setup.py or any in-dir builds

##### also KGF TODO pip DEPRECATION: --build-option and --global-option are deprecated. pip 23.3 will enforce this behaviour change.

#DS_BUILD_EVOFORMER_ATTN=0 ????
# Evoformer (and its CUTLASS dependency) was not added to master until AFTER 0.10.3 (you can grep the entire repo for it, and it doesnt appear until then)
# The op wont appear in ds_report unless building unstable master

#DS_BUILD_UTILS no longer does anything after 0.9.x: https://github.com/microsoft/DeepSpeed/issues/4422

# CMAKE_POSITION_INDEPENDENT_CODE=ON NVCC_PREPEND_FLAGS="--forward-unknown-opts"  # https://github.com/microsoft/DeepSpeed/issues/3233
# KGF: the PIC option seems to cause problems, but maybe the latter option helps? nvcc was emitting warnings about -Wno-reorder
# the following line works with 0.10.3
# NVCC_PREPEND_FLAGS="--forward-unknown-opts" DS_BUILD_SPARSE_ATTN=0 DS_BUILD_OPS=1 DS_BUILD_AIO=1 pip install --verbose deepspeed


cd $BASE_PATH
echo "Install DeepSpeed from source"
git clone https://github.com/microsoft/DeepSpeed.git
cd DeepSpeed
# HARDCODE
git checkout v0.10.3
export CFLAGS="-I${CONDA_PREFIX}/include/"
export LDFLAGS="-L${CONDA_PREFIX}/lib/ -Wl,--enable-new-dtags,-rpath,${CONDA_PREFIX}/lib"
# KGF: above two lines need to be added to modulefile?

#DS_BUILD_SPARSE_ATTN=0 DS_BUILD_OPS=1 DS_BUILD_AIO=1 DS_BUILD_UTILS=1 bash install.sh --verbose

#NVCC_PREPEND_FLAGS="--forward-unknown-opts" DS_BUILD_SPARSE_ATTN=0 DS_BUILD_OPS=1 DS_BUILD_AIO=1 pip install --verbose . --global-option="build_ext" --global-option="-j16"
# ERROR:  /soft/datascience/conda/2023-09-29/mconda3/compiler_compat/ld: build/temp.linux-x86_64-cpython-310/csrc/transformer/normalize_kernels.o: error adding symbols: no error

NVCC_PREPEND_FLAGS="--forward-unknown-opts" DS_BUILD_SPARSE_ATTN=0 DS_BUILD_OPS=1 DS_BUILD_AIO=1 pip install --verbose .  # PASSES with 0.10.3

#NVCC_PREPEND_FLAGS="--forward-unknown-opts" DS_BUILD_SPARSE_ATTN=0 DS_BUILD_OPS=1 DS_BUILD_AIO=1 pip install --verbose . ### FAILS with master
# /soft/datascience/conda/2023-10-01/mconda3/compiler_compat/ld: build/temp.linux-x86_64-cpython-310/csrc/deepspeed4science/evoformer_attn/attention.o:(.bss+0x0): multiple definition of `torch::autograd::(anonymous namespace)::graph_task_id'; build/temp.linux-x86_64-cpython-310/csrc/deepspeed4science/evoformer_attn/attention.o:(.bss+0x0): first defined here


#DS_BUILD_SPARSE_ATTN=0 DS_BUILD_OPS=1 DS_BUILD_AIO=1 pip install --verbose deepspeed --global-option="build_ext" --global-option="-j32"

#NVCC_PREPEND_FLAGS="--forward-unknown-opts" DS_BUILD_SPARSE_ATTN=0 DS_BUILD_OPS=1 DS_BUILD_AIO=1 python setup.py build_ext -j16 bdist_wheel

# ---> error: command '/soft/compilers/cudatoolkit/cuda-11.8.0/bin/nvcc' failed with exit code 1 (real error seems hidden?)

# if no rpath, add this to Lmod modulefile:
#export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${CONDA_PREFIX}/lib"
# Suboptimal, since we really only want "libaio.so" from that directory to run DeepSpeed.
# e.g. if you put "export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}", it overrides many system libraries
# breaking "module list", "emacs", etc.

# > ds_report
cd $BASE_PATH

# HARDCODE
###pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
# --- Building wheels for collected packages: nvidia-cublas-cu11, nvidia-cuda-cupti-cu11, nvidia-cuda-nvcc-cu11, nvidia-cuda-runtime-cu11, nvidia-cudnn-cu11

# KGF POTENTIAL ISSUE SEPTEMBER 2023:
# Try building from source, OR getting wheel that links to local CUDA and cuDNN
# See https://jax.readthedocs.io/en/latest/developer.html#building-from-source
# OR
pip install --upgrade "jax[cuda11_local]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
#
# CONCERN: Some GPU functionality expects the CUDA installation to be at /usr/local/cuda-X.X, where X.X should be replaced with the CUDA version number (e.g. cuda-11.8). If CUDA is installed elsewhere on your system, you can either create a symlink:
# -----
# jax[cuda11_pip] brings:
# Installing collected packages: nvidia-cudnn-cu116, nvidia-cuda-runtime-cu117, nvidia-cuda-nvcc-cu117, nvidia-cuda-cupti-cu117, nvidia-cublas-cu117, ml-dtypes, nvidia-cudnn-cu11, nvidia-cuda-runtime-cu11, nvidia-cuda-nvcc-cu11, nvidia-cuda-cupti-cu11, nvidia-cublas-cu11, jaxlib, jax
#   Attempting uninstall: nvidia-cudnn-cu11
#     Found existing installation: nvidia-cudnn-cu11 8.5.0.96
#     Uninstalling nvidia-cudnn-cu11-8.5.0.96:
#       Successfully uninstalled nvidia-cudnn-cu11-8.5.0.96
#   Attempting uninstall: nvidia-cuda-runtime-cu11
#     Found existing installation: nvidia-cuda-runtime-cu11 11.7.99
#     Uninstalling nvidia-cuda-runtime-cu11-11.7.99:
#       Successfully uninstalled nvidia-cuda-runtime-cu11-11.7.99
#   Attempting uninstall: nvidia-cuda-cupti-cu11
#     Found existing installation: nvidia-cuda-cupti-cu11 11.7.101
#     Uninstalling nvidia-cuda-cupti-cu11-11.7.101:
#       Successfully uninstalled nvidia-cuda-cupti-cu11-11.7.101
#   Attempting uninstall: nvidia-cublas-cu11
#     Found existing installation: nvidia-cublas-cu11 11.10.3.66
#     Uninstalling nvidia-cublas-cu11-11.10.3.66:
#       Successfully uninstalled nvidia-cublas-cu11-11.10.3.66
# ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
# torch 2.0.1 requires nvidia-cublas-cu11==11.10.3.66; platform_system == "Linux" and platform_machine == "x86_64", but you have nvidia-cublas-cu11 2022.4.8 which is incompatible.
# torch 2.0.1 requires nvidia-cuda-cupti-cu11==11.7.101; platform_system == "Linux" and platform_machine == "x86_64", but you have nvidia-cuda-cupti-cu11 2022.4.8 which is incompatible.
# torch 2.0.1 requires nvidia-cuda-runtime-cu11==11.7.99; platform_system == "Linux" and platform_machine == "x86_64", but you have nvidia-cuda-runtime-cu11 2022.4.25 which is incompatible.
# torch 2.0.1 requires nvidia-cudnn-cu11==8.5.0.96; platform_system == "Linux" and platform_machine == "x86_64", but you have nvidia-cudnn-cu11 2022.5.19 which is incompatible.
# Successfully installed jax-0.4.16 jaxlib-0.4.16+cuda11.cudnn86 ml-dtypes-0.3.0 nvidia-cublas-cu11-2022.4.8 nvidia-cublas-cu117-11.10.1.25 nvidia-cuda-cupti-cu11-2022.4.8 nvidia-cuda-cupti-cu117-11.7.50 nvidia-cuda-nvcc-cu11-2022.5.4 nvidia-cuda-nvcc-cu117-11.7.64 nvidia-cuda-runtime-cu11-2022.4.25 nvidia-cuda-runtime-cu117-11.7.60 nvidia-cudnn-cu11-2022.5.19 nvidia-cudnn-cu116-8.4.0.27

#pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
#pip install "jax[cuda11_cudnn86]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install pymongo optax flax
pip install "numpyro[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html


# --- MPI4JAX
# https://github.com/mpi4jax/mpi4jax/issues/153
# CUDA_ROOT=/soft/datascience/cuda/cuda_11.5.2_495.29.05_linux python setup.py --verbose build_ext --inplace
# be sure to "rm -rfd build/" to force .so libraries to rebuild if you change the build options, etc.

#git clone https://github.com/argonne-lcf/mpi4jax.git
pip install cython
git clone https://github.com/mpi4jax/mpi4jax.git
cd mpi4jax
#git checkout polaris
CUDA_ROOT=$CUDA_TOOLKIT_BASE pip install --no-build-isolation --no-cache-dir --no-binary=mpi4jax -v .
cd $BASE_PATH

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

echo "Cleaning up"
chmod -R u+w $DOWNLOAD_PATH/
rm -rf $DOWNLOAD_PATH

# KGF: see below
conda list

chmod -R a-w $BASE_PATH/


set +e
