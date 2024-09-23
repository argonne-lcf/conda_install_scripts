#!/bin/bash -l

# As of May 2022
# This script will install TensorFlow, PyTorch, and Horovod on Polaris, all from source
# 1 - Login to Polaris login-node
# 2 - Run './<this script> /path/to/install/base/'
# 3 - script installs everything down in /path/to/install/base/
# 4 - wait for it to complete

# KGF: check HARDCODE points for lines that potentially require manual edits to pinned package versions

# e.g. /lus/theta-fs0/software/thetagpu/conda/2023-01-11 on ThetaGPU, /soft/datascience/conda/2023-01-10 on Polaris
BASE_PATH=$1
DATE_PATH="$(basename $BASE_PATH)"

export PYTHONNOUSERSITE=1
# KGF: PBS used to mess with user umask, changing it to 0077 on compute node
# dirs that were (2555/dr-xr-sr-x) on ThetaGPU became (2500/dr-x--S---)
umask 0022

# move primary conda packages directory/cache away from ~/.conda/pkgs (4.2 GB currently)
# hardlinks should be preserved even if these files are moved (not across filesystem boundaries)
export CONDA_PKGS_DIRS=/soft/datascience/conda/pkgs

# https://stackoverflow.com/questions/67610133/how-to-move-conda-from-one-folder-to-another-at-the-moment-of-creating-the-envi
# > conda config --get
# --add envs_dirs '/lus/theta-fs0/projects/fusiondl_aesp/conda/envs'
# --set pip_interop_enabled False
# --add pkgs_dirs '$HOME/.conda/pkgs'
# --add pkgs_dirs '/lus/theta-fs0/projects/fusiondl_aesp/conda/pkgs'

# > conda info --root
# /soft/datascience/conda/2022-07-19-login/mconda3

# If the pkgs_dirs key is not set, then envs/pkgs is used as the pkgs cache, except for
# the standard envs directory in the root directory, for which the normal root_dir/pkgs is
# used.
# https://docs.conda.io/projects/conda/en/latest/user-guide/configuration/use-condarc.html#specify-package-directories-pkgs-dirs

# https://docs.conda.io/projects/conda/en/latest/configuration.html
# #   The list of directories where locally-available packages are linked
# #   from at install time. Packages not locally available are downloaded
# #   and extracted into the first writable directory.
# #
# pkgs_dirs: []


# Note, /soft and /home currently (temporarily) share a filesystem as of July 2022
# Default 100 GB quota will be exhauted quickly.

#########################################################
# Check for outside communication on ThetaGPU
# (be sure not to inherit these vars from dotfiles)
###########
unset https_proxy
unset http_proxy

wget -q --spider -T 10 http://google.com
if [ $? -eq 0 ]; then
    echo "Network Online"
else
    # non-/interactive full-node job without --attrs=pubnet on ThetaGPU
    echo "Network Offline, setting proxy envs"
    export https_proxy=http://proxy.alcf.anl.gov:3128
    export http_proxy=http://proxy.alcf.anl.gov:3128
fi

#set -e

# PrgEnv-nvhpc is default PE on Perlmutter as of 2022-07-27
module list
# ----------------------
# See Polaris module pre- vs. post-AT notes: https://github.com/felker/athenak-scaling/blob/main/README.md
# Change for Prgenv-nvidia and Nvidia Module Files: As pre-announced with the CPE-HPCM
# 22.02 release and starting with the CPE 22.03 release, the PrgEnv-nvidia and nvidia
# module files are being deprecated in favor of PrgEnv-nvhpc and nvhpc, respectively, and
# may be removed in a later release.

# During AT, Lmod was not used. PrgEnv-nvidia was default.
# Perlmutter only documents PrgEnv-nvidia, since PrgEnv-nvhpc came later
# The Tcl modulefiles were very similar. Only differences, which made
# the following error out:
# module switch PrgEnv-nvidia/8.3.3 PrgEnv-nvhpc
# nvidia/22.3(6):ERROR:150: Module 'nvidia/22.3' conflicts with the currently loaded module(s) 'nvhpc/22.3'

# PrgEnv-nvidia
# setenv           nvidia_already_loaded 1
# module           swap nvidia/22.3

# PrgEnv-nvhpc
# setenv           nvhpc_already_loaded 0
# module           load nvhpc
# ----------------------------
#module switch PrgEnv-nvidia PrgEnv-gnu
# second module is the final module. First one is unloaded
# note "switch" and "swap" are aliases in both Env Modules and Lmod

module load PrgEnv-nvhpc  # not actually using NVHPC compilers to build TF
#module load PrgEnv-gnu
module load gcc-mixed # get 12.2.0 (2022) instead of /usr/bin/gcc 7.5 (2019)
module load craype-accel-nvidia80  # wont load for PrgEnv-gnu; see HPE Case 5367752190
module load craype-x86-milan
export MPICH_GPU_SUPPORT_ENABLED=1
module list
echo $MPICH_DIR

# -------------------- begin HARDCODE of major built-from-source frameworks etc.
# unset *_TAG variables to build latest master/main branch (or "develop" in the case of DeepHyper)
#DH_REPO_TAG="0.4.2"
DH_REPO_URL=https://github.com/deephyper/deephyper.git

TF_REPO_TAG="v2.16.1"
PT_REPO_TAG="v2.3.0"
HOROVOD_REPO_TAG="v0.28.1" # e.g. v0.22.1 released on 2021-06-10 should be compatible with TF 2.6.x and 2.5.x
TF_REPO_URL=https://github.com/tensorflow/tensorflow.git
HOROVOD_REPO_URL=https://github.com/uber/horovod.git
PT_REPO_URL=https://github.com/pytorch/pytorch.git

# MPI4PY_REPO_URL=https://github.com/mpi4py/mpi4py.git
# MPI4PY_REPO_TAG="3.1.3"
# -------------------- end HARDCODE of major built-from-source frameworks etc.

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
CUDA_VERSION_MINOR=4
CUDA_VERSION_MINI=1
#CUDA_VERSION_MINOR=2
#CUDA_VERSION_MINI=0

#CUDA_VERSION_BUILD=495.29.05
CUDA_VERSION=$CUDA_VERSION_MAJOR.$CUDA_VERSION_MINOR
CUDA_VERSION_FULL=$CUDA_VERSION.$CUDA_VERSION_MINI

CUDA_TOOLKIT_BASE=/soft/compilers/cudatoolkit/cuda-${CUDA_VERSION_FULL}
CUDA_HOME=${CUDA_TOOLKIT_BASE}

CUDA_DEPS_BASE=/soft/libraries/

CUDNN_VERSION_MAJOR=9
CUDNN_VERSION_MINOR=1
CUDNN_VERSION_EXTRA=0.70
CUDNN_VERSION=$CUDNN_VERSION_MAJOR.$CUDNN_VERSION_MINOR.$CUDNN_VERSION_EXTRA
#CUDNN_BASE=$CUDA_DEPS_BASE/cudnn/cudnn-$CUDA_VERSION-linux-x64-v$CUDNN_VERSION
# HARDCODE: manually renaming default cuDNN tarball name to fit this schema:
CUDNN_BASE=$CUDA_DEPS_BASE/cudnn/cudnn-cuda$CUDA_VERSION_MAJOR-linux-x64-v$CUDNN_VERSION

NCCL_VERSION_MAJOR=2
NCCL_VERSION_MINOR=21.5-1
NCCL_VERSION=$NCCL_VERSION_MAJOR.$NCCL_VERSION_MINOR
NCCL_BASE=$CUDA_DEPS_BASE/nccl/nccl_$NCCL_VERSION+cuda${CUDA_VERSION}_x86_64
# KGF: no Extended Compatibility in NCCL --- use older NCCL version built with earlier CUDA version until
# GPU device kernel driver is upgraded

# https://github.com/tensorflow/tensorflow/pull/55634

# https://docs.nvidia.com/deeplearning/tensorrt/support-matrix/index.html

# TRT 10 is still EA in April 2024; doesnt seem fully supported by TensorFlow 2.16.1
# (e.g. include/NvUtils.h is gone; was deprecated in 9.0.x, but should have kept the deprecated API for some time??)
# https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-1000-ea/release-notes/index.html#rel-10-0-0-EA

# Last few releases of TensorRT were tested with TF 2.12.0
# https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-1000-ea/release-notes/index.html

# TRT 9 closed source multi-gigabyte downloads not available on NVIDIA's website:
# https://github.com/NVIDIA/TensorRT/issues/3741
# > I think it's expected since TRT 9 is a limited release that didn't go public in our developer zone.
# > P.S. we just release TRT 10 to public.

TENSORRT_VERSION_MAJOR=8
TENSORRT_VERSION_MINOR=6.1.6

# TENSORRT_VERSION_MAJOR=10
# TENSORRT_VERSION_MINOR=0.0.6
TENSORRT_VERSION=$TENSORRT_VERSION_MAJOR.$TENSORRT_VERSION_MINOR
# https://github.com/tensorflow/tensorflow/pull/55634
#TENSORRT_BASE=$CUDA_DEPS_BASE/trt/TensorRT-$TENSORRT_VERSION.Linux.x86_64-gnu.cuda-$CUDA_VERSION.cudnn$CUDNN_VERSION_MAJOR.$CUDNN_VERSION_MINOR
# TensorRT-8.6.1.6.Linux.x86_64-gnu.cuda-11.8.cudnn8.9/
#TENSORRT_BASE=$CUDA_DEPS_BASE/trt/TensorRT-$TENSORRT_VERSION.Linux.x86_64-gnu.cuda-$CUDA_VERSION.cudnn$CUDNN_VERSION_MAJOR.9

# TensorRT-8.5.3.1.Linux.x86_64-gnu.cuda-11.8.cudnn8.6/
#TENSORRT_BASE=$CUDA_DEPS_BASE/trt/TensorRT-$TENSORRT_VERSION.Linux.x86_64-gnu.cuda-$CUDA_VERSION.cudnn$CUDNN_VERSION_MAJOR.6

# TensorRT-8.6.1.6.Linux.x86_64-gnu.cuda-11.8.cudnn8.9/ and cuDNN 8.7.0.84: fails TensorFlow building
# bazel-out/k8-opt-exec-50AE0418/bin/external/local_config_tensorrt/_virtual_includes/tensorrt_headers/third_party/tensorrt/NvInferRuntimeCommon.h:26:10: fatal error: NvInferRuntimeBase.h: No such file or directory

# TensorRT-10.0.0.6.Linux.x86_64-gnu.cuda-12.4/

# KGF: no cuDNN compat specified? Where was this documented in early versions?
# in 10.x.x release notes:
# > cuDNN is now an optional dependency for TensorRT and is only used to speed-up a small number of layers.
# >  TensorRT 10.0.0 Early Access supports cuDNN 8.9.7. cuDNN is not used by the lean or dispatch runtimes.

# in 8.6.1 release notes:
# > The tar and zip filenames no longer contain the cuDNN version due to cuDNN no longer being a primary tactic source for TensorRT. cuDNN has a lesser impact on TensorRT’s performance than in previous releases where the cuDNN version was prominent in the package filename.

# TensorRT-8.6.1.6.Linux.x86_64-gnu.cuda-12.0
# HARDCODE
TENSORRT_BASE=$CUDA_DEPS_BASE/trt/TensorRT-$TENSORRT_VERSION.Linux.x86_64-gnu.cuda-12.0

echo "TENSORRT_BASE=${TENSORRT_BASE}"

# is the following trt compatible with cuDNN 8.7 too?
# TENSORRT_BASE=$CUDA_DEPS_BASE/trt/TensorRT-$TENSORRT_VERSION.Linux.x86_64-gnu.cuda-$CUDA_VERSION.cudnn$CUDNN_VERSION_MAJOR.6


#################################################
# TensorFlow Config flags (for ./configure run)
#################################################
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
# KGF: TensorRT 8.x only supported in TensorFlow as of 2021-06-24 (f8e2aa0db)
# https://github.com/tensorflow/tensorflow/issues/49150
# https://github.com/tensorflow/tensorflow/pull/48917
export TF_NEED_TENSORRT=1
export TF_CUDA_PATHS=$CUDA_TOOLKIT_BASE,$CUDNN_BASE,$NCCL_BASE,$TENSORRT_BASE
#export GCC_HOST_COMPILER_PATH=$(which gcc)
# HARDCODE
export TF_PYTHON_VERSION=3.11
export GCC_HOST_COMPILER_PATH=/opt/cray/pe/gcc/12.2.0/snos/bin/gcc
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
CONDAVER='py311_24.3.0-0'
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

#########
# create a setup file
cat > setup.sh << EOF
preferred_shell=\$(basename \$SHELL)

module load PrgEnv-gnu
#module load PrgEnv-nvhpc

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


PYTHON_VER=$(ls -d lib/python?.? | tail -c4)
echo PYTHON_VER=$PYTHON_VER

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

# note, numba pulls in numpy here too
conda install -y -c defaults -c conda-forge cmake zip unzip astunparse setuptools future six requests dataclasses graphviz numba numpy pymongo conda-build pip libaio
conda install -y -c defaults -c conda-forge mkl mkl-include  # onednn mkl-dnn git-lfs ### on ThetaGPU
# conda install -y cffi typing_extensions pyyaml

# KGF: note, ordering of the above "defaults" channel install relative to "conda install -y -c conda-forge mamba; conda install -y pip"
# (used to leave the pip re-install on a separate line) may affect what version of numpy you end up with
# E.g. Jan 2023, Polaris ordering (defaults, then mamba then pip) got numpy 1.23.5 and ThetaGPU (mamba, pip, then defaults) got numpy 1.21.5

# CUDA only: Add LAPACK support for the GPU if needed
# HARDCODE
conda install -y -c defaults -c pytorch -c conda-forge magma-cuda${CUDA_VERSION_MAJOR}${CUDA_VERSION_MINOR}
# KGF(2022-09-13): note, if you were to explicitly specifying conda-forge channel here but not in the global or local .condarc list of channels set, it would cause issues with cloned environments being unable to download the package
conda install -y -c defaults -c conda-forge mamba ccache
# KGF: mamba is not on "defaults" channel, and no easy way to build from source via pip since it is a full
# package manager, not just a Python module, etc.
# - might not need to explicitly pass "-c conda-forge" now that .condarc is updated
# - should I "conda install -y -c defaults -c conda-forge mamba" so that dep packages follow same channel precedence as .condarc? doesnt seem to matter--- all ~4x deps get pulled from conda-forge

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
BAZEL_VERSION=$(head -n1 .bazelversion)
echo "Found TensorFlow depends on Bazel version $BAZEL_VERSION"

cd $BASE_PATH
echo "Download Bazel binaries"
BAZEL_DOWNLOAD_URL=https://github.com/bazelbuild/bazel/releases/download/$BAZEL_VERSION
BAZEL_INSTALL_SH=bazel-$BAZEL_VERSION-installer-linux-x86_64.sh
BAZEL_INSTALL_PATH=$BASE_PATH/bazel-$BAZEL_VERSION
echo "wget $BAZEL_DOWNLOAD_URL/$BAZEL_INSTALL_SH -P $DOWNLOAD_PATH"
wget $BAZEL_DOWNLOAD_URL/$BAZEL_INSTALL_SH -P $DOWNLOAD_PATH
chmod +x $DOWNLOAD_PATH/$BAZEL_INSTALL_SH
echo "Install Bazel in $BAZEL_INSTALL_PATH"
bash $DOWNLOAD_PATH/$BAZEL_INSTALL_SH --prefix=$BAZEL_INSTALL_PATH
export PATH=$PATH:/$BAZEL_INSTALL_PATH/bin

cd $BASE_PATH

echo "Install TensorFlow Dependencies"
#pip install -U pip six 'numpy<1.19.0' wheel setuptools mock 'future>=0.17.1' 'gast==0.3.3' typing_extensions portpicker
# KGF: try relaxing the dependency version requirements (esp NumPy, since PyTorch wants a later version?)
#pip install -U pip six 'numpy~=1.19.5' wheel setuptools mock future gast typing_extensions portpicker pydot
# KGF (2021-12-15): stop limiting NumPy for now. Unclear if problems with 1.20.3 and TF/Pytorch
pip install -U numpy numba ninja
# the above line can be very important or very bad, to get have pip control the numpy dependency chain right before TF build
# Start including numba here too in order to ensure mutual compat; numba 0.56.4 req numpy <1.24.0, e.g.
# Check https://github.com/numpy/numpy/blob/main/numpy/core/setup_common.py
# for C_API_VERSION, and track everytime numpy is reinstalled in the build log
pip install -U pip wheel mock gast portpicker pydot packaging pyyaml
pip install -U keras_applications --no-deps
pip install -U keras_preprocessing --no-deps

echo "Configure TensorFlow"
cd tensorflow
export PYTHON_BIN_PATH=$(which python)
export PYTHON_LIB_PATH=$(python -c 'import site; print(site.getsitepackages()[0])')
# Auto-Configuration Warning: 'TMP' environment variable is not set, using 'C:\Windows\Temp' as default
export TMP=/tmp
./configure

# was getting an error related to tensorflow trying to call `/opt/cray/pe/gcc/11.2.0/bin/redirect` directly
# however, this redirect is a bash script in the Cray PE GCC
# `/opt/cray/pe/gcc/11.2.0/bin/gcc` and the other compiler commands in that folder are all symlinks
# to the redirect script which simply replaces the base path in the command with the true location of the
# commands which were in `/opt/cray/pe/gcc/11.2.0/bin/../snos/bin`
# `redirect` is not intended to be called directly.
# However, the tensorflow build environment saw that `gcc` was  symlink and dereferenced it to set:
# GCC_HOST_COMPILER_PATH=/opt/cray/pe/gcc/11.2.0/bin/redirect
# at compile time, which fails. So we instead fix the gcc to use this:
# KGF: see above, around L180

echo "Bazel Build TensorFlow"
# KGF: restrict Bazel to only see 32 cores of the dual socket 64-core (physical) AMD Epyc node (e.g. 256 logical cores)
# Else, Bazel will hit PID limit, even when set to 32,178 in /sys/fs/cgroup/pids/user.slice/user-XXXXX.slice/pids.max
# even if --jobs=500
HOME=$DOWNLOAD_PATH bazel build --jobs=500 --local_cpu_resources=32 --verbose_failures --config=cuda --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0" //tensorflow/tools/pip_package:build_pip_package
echo "Run wheel building"
./bazel-bin/tensorflow/tools/pip_package/build_pip_package $WHEELS_PATH
echo "Install TensorFlow"
pip install $(find $WHEELS_PATH/ -name "tensorflow*.whl" -type f)


#################################################
### Install PyTorch
#################################################

cd $BASE_PATH
echo "Clone PyTorch"

# KGF: 2023-09, --recursive clone caused issues in subsequent command "git checkout --recurse-submodules" (likely this one)
# or "git submodule sync" within script on Polaris (but couldnt reproduce on login node interactively, nor on my macOS system:
# "fatal: not a git repository: ../../.git/modules/third_party/python-enum"

# https://github.com/pytorch/pytorch/tree/release/2.0#get-the-pytorch-source
# still recommends those sequence of commands, except the branch checkout
# Maybe the --recurse-submodules is an issue if a submodule is undefined on the target branch?

# git clone --recursive $PT_REPO_URL
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
# v2.0.1 (redundant) 2x fixes: https://github.com/pytorch/pytorch/issues/107389
#echo "git apply patch for v2.0.1"
#git apply ~/hardcode_aten_cudnn.patch
export CUDNN_INCLUDE_DIR=$CUDNN_BASE/include
export CPATH="$CPATH:$CUDNN_INCLUDE_DIR"

echo "Install PyTorch"
module unload gcc-mixed
module load PrgEnv-gnu
# April 2024 Sirius: you end up with gcc-native/12.3, PrgEnv-gnu/8.5.0, which despite the earlier modulefile:
# ❯ which gcc
# /usr/bin/gcc
# ❯ gcc --version
# gcc (SUSE Linux) 7.5.0

####module load gcc/12.2.0
# ❯ which gcc
# /opt/cray/pe/gcc/12.2.0/snos/bin/gcc
# ❯ which CC
# /opt/cray/pe/craype/2.7.30/bin/CC
# ❯ which cc
# /opt/cray/pe/craype/2.7.30/bin/cc

# KGF: wont load due to modulefile: prereq_any(atleast("cudatoolkit","11.0"), "nvhpc", "PrgEnv-nvhpc")
# need to relax this or change to prereq_any(atleast("cudatoolkit-standalone","11.0"), "nvhpc", "PrgEnv-nvhpc")
# KGF: any way to "module load --force"???
#module load craype-accel-nvidia80
export CRAY_ACCEL_TARGET="nvidia80"
export CRAY_TCMALLOC_MEMFS_FORCE="1"
export CRAYPE_LINK_TYPE="dynamic"
export CRAY_ACCEL_VENDOR="nvidia"
export CRAY_CPU_TARGET="x86-64"
#### export CRAY_CPU_TARGET="craype-x86-milan"  # dont do this

module list
echo "CRAY_ACCEL_TARGET= $CRAY_ACCEL_TARGET"
echo "CRAYPE_LINK_TYPE = $CRAYPE_LINK_TYPE"

export USE_CUDA=1
export USE_CUDNN=1
export TORCH_CUDA_ARCH_LIST=8.0
#export CUDA_TOOLKIT_ROOT_DIR=$CUDA_TOOLKIT_BASE
#export CUDA_HOME=$CUDA_TOOLKIT_BASE
#export NCCL_ROOT_DIR=$NCCL_BASE
export CUDNN_ROOT=$CUDNN_BASE
echo "CUDNN_ROOT=$CUDNN_BASE"
export CUDNN_ROOT_DIR=$CUDNN_BASE
export CUDNN_INCLUDE_DIR=$CUDNN_BASE/include

export USE_TENSORRT=ON
export TENSORRT_ROOT=$TENSORRT_BASE
export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
#export TENSORRT_LIBRARY=$TENSORRT_BASE/lib/libmyelin.so
#export TENSORRT_LIBRARY_INFER=$TENSORRT_BASE/lib/libnvinfer.so
#export TENSORRT_LIBRARY_INFER_PLUGIN=$TENSORRT_BASE/lib/libnvinfer_plugin.so
#export TENSORRT_INCLUDE_DIR=$TENSORRT_BASE/include

# -------------
# KGF 2023-09 update: torch devs never update version.txt for patch releases, but it is dilligently
# uploaded before a minor release is tagged, e.g. compare:
# https://github.com/pytorch/pytorch/blob/v1.13.0/version.txt
# https://github.com/pytorch/pytorch/blob/v1.13.1/version.txt
# https://github.com/pytorch/pytorch/blob/release/1.13/version.txt
# etc. In fact, they never mint more than 1x patch release per minor version

# When building wheels for distribution, they likely always override version.txt via below 2x env vars.
# Indeed, they do: https://github.com/pytorch/builder/blob/main/wheel/build_wheel.sh

# see setup.py:
#   PYTORCH_BUILD_VERSION
#   PYTORCH_BUILD_NUMBER
#     specify the version of PyTorch, rather than the hard-coded version
#     in this file; used when we're building binaries for distribution

# "version" variable passed to setup() is from get_torch_version() in tools/generate_torch_version.py
# which parses those env vars, or falls back to version.txt

# https://discuss.pytorch.org/t/how-to-build-from-source-code-without-trailing-torch-1-8-0a0-gitf2a38a0-version/115698/4
# https://discuss.pytorch.org/t/how-to-specify-pytorch-version-with-the-cuda-version-when-building-from-source/134447
# https://github.com/pytorch/pytorch/issues/50730#issuecomment-763207634
# https://github.com/pytorch/pytorch/issues/61468
# https://github.com/pytorch/pytorch/issues/20525
# https://github.com/pytorch/pytorch/issues/51868
# > More that version.txt is only used in the absence of PYTORCH_BUILD_VERSION which means that it actually never gets updated.

# https://github.com/pytorch/pytorch/pull/95790#issuecomment-1450443579
# (on why they are bumping main's version.txt to 2.1.0a0 even though 2.0.0 hasnt been released as of 2023-03-01)
# > Release 2.0.0 is coming. We have a branch for it, release/2.0 . Normally we should bump the version as soon as we do the branch

# https://github.com/pytorch/pytorch/issues/9926
# https://github.com/pytorch/pytorch/issues/7954#issuecomment-394443358

# scripts/release/cut-release-branch.sh script is just for creating release/X.Y branches
# and pushing them to remotes. It does read from version.txt:
# RELEASE_VERSION=${RELEASE_VERSION:-$(cut -d'.' -f1-2 "${GIT_TOP_DIR}/version.txt")}

# https://github.com/pytorch/pytorch/commits/release/2.0
# e.g. tag v2.0.1 is on this branch, and there are even 3x commits beyond that even though there isnt yet a 2.0.2
# See https://github.com/pytorch/pytorch/blob/main/RELEASE.md#pytorchbuilder--pytorch-domain-libraries
# (trim the first character of the tag, "v")
export PYTORCH_BUILD_VERSION="${PT_REPO_TAG:1}"
export PYTORCH_BUILD_NUMBER=1

export TENSORRT_INCLUDE_DIR="TENSORRT_BASE/include"
export TENSORRT_LIBRARY="$TENSORRT_BASE/lib/libmyelin.so"
#export TENSORRT_LIBRARY="$TENSORRT_BASE/lib"
# cuSPARSELt added in PyTorch 2.1 https://github.com/pytorch/pytorch/issues/62153
export USE_CUSPARSELT=1
export CUSPARSELT_ROOT="/soft/libraries/cusparselt/libcusparse_lt-linux-x86_64-0.6.0.6/"
export CUSPARSELT_INCLUDE_PATH="/soft/libraries/cusparselt/libcusparse_lt-linux-x86_64-0.6.0.6/include"
# -------------

echo "TENSORRT_INCLUDE_DIR=${TENSORRT_INCLUDE_DIR}"
echo "TENSORRT_LIBRARY=${TENSORRT_LIBRARY}"
echo "CUSPARSELT_ROOT=${CUSPARSELT_ROOT}"
echo "CUSPARSELT_INCLUDE_PATH=${CUSPARSELT_INCLUDE_PATH}"

echo "PYTORCH_BUILD_VERSION=$PYTORCH_BUILD_VERSION and PYTORCH_BUILD_NUMBER=$PYTORCH_BUILD_NUMBER"
#echo "CC=/opt/cray/pe/gcc/12.2.0/snos/bin/gcc CXX=/opt/cray/pe/gcc/12.2.0/snos/bin/g++ python setup.py bdist_wheel"
echo "BUILD_TEST=0 CUDAHOSTCXX=g++-12 CC=/usr/bin/gcc-12 CXX=/usr/bin/g++-12 python setup.py bdist_wheel"
#echo "CC=$(which cc) CXX=$(which CC) python setup.py bdist_wheel"
#CC=$(which cc) CXX=$(which CC) python setup.py bdist_wheel
# HARDCODE
BUILD_TEST=0 CUDAHOSTCXX=g++-12 CC=/usr/bin/gcc-12 CXX=/usr/bin/g++-12 python setup.py bdist_wheel
PT_WHEEL=$(find dist/ -name "torch*.whl" -type f)
echo "copying pytorch wheel file $PT_WHEEL"
cp $PT_WHEEL $WHEELS_PATH/
cd $WHEELS_PATH
echo "pip installing $(basename $PT_WHEEL)"
pip install $(basename $PT_WHEEL)
# HARDCODE
#pip install torchtriton --extra-index-url "https://download.pytorch.org/whl/nightly/cu124"
# https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html

################################################
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

echo "Build Horovod Wheel using MPI from $MPICH_DIR and NCCL from ${NCCL_BASE}"
# https://horovod.readthedocs.io/en/stable/gpus_include.html
# If you installed NCCL 2 using the nccl-<version>.txz package, you should specify the path to NCCL 2 using the HOROVOD_NCCL_HOME environment variable.
# add the library path to LD_LIBRARY_PATH environment variable or register it in /etc/ld.so.conf.
#export LD_LIBRARY_PATH=$CRAY_MPICH_PREFIX/lib-abi-mpich:$NCCL_BASE/lib:$LD_LIBRARY_PATH
#export PATH=$CRAY_MPICH_PREFIX/bin:$PATH

# https://github.com/horovod/horovod/issues/3696#issuecomment-1248921736
echo "CUDAHOSTCXX=g++-12 CC=/usr/bin/gcc-12 CXX=/usr/bin/g++-12 HOROVOD_WITH_MPI=1 HOROVOD_CUDA_HOME=${CUDA_TOOLKIT_BASE} HOROVOD_NCCL_HOME=$NCCL_BASE HOROVOD_CMAKE=$(which cmake) HOROVOD_GPU_OPERATIONS=NCCL HOROVOD_WITH_TENSORFLOW=1 HOROVOD_WITHOUT_PYTORCH=1 HOROVOD_WITHOUT_MXNET=1 python setup.py bdist_wheel"

# HARDCODE: temp disable Horovod 0.28.1 + PyTorch >=2.1.x integration; C++17 required in PyTorch now (https://github.com/pytorch/pytorch/pull/100557)
# https://github.com/horovod/horovod/pull/3998
# https://github.com/horovod/horovod/issues/3996
CUDAHOSTCXX=g++-12 CC=/usr/bin/gcc-12 CXX=/usr/bin/g++-12 HOROVOD_WITH_MPI=1 HOROVOD_CUDA_HOME=${CUDA_TOOLKIT_BASE} HOROVOD_NCCL_HOME=$NCCL_BASE HOROVOD_CMAKE=$(which cmake) HOROVOD_GPU_OPERATIONS=NCCL HOROVOD_WITH_TENSORFLOW=1 HOROVOD_WITHOUT_PYTORCH=1 HOROVOD_WITHOUT_MXNET=1 python setup.py bdist_wheel
#CUDAHOSTCXX=g++-12 CC=/usr/bin/gcc-12 CXX=/usr/bin/g++-12 HOROVOD_WITH_MPI=1 HOROVOD_CUDA_HOME=${CUDA_TOOLKIT_BASE} HOROVOD_NCCL_HOME=$NCCL_BASE HOROVOD_CMAKE=$(which cmake) HOROVOD_GPU_OPERATIONS=NCCL HOROVOD_WITH_TENSORFLOW=1 HOROVOD_WITH_PYTORCH=1 HOROVOD_WITHOUT_MXNET=1 python setup.py bdist_wheel
# HOROVOD_GPU_ALLREDUCE=MPI, HOROVOD_GPU_OPERATIONS=MPI

HVD_WHL=$(find dist/ -name "horovod*.whl" -type f)
cp $HVD_WHL $WHEELS_PATH/
HVD_WHEEL=$(find $WHEELS_PATH/ -name "horovod*.whl" -type f)
echo "Install Horovod $HVD_WHEEL"
pip install --force-reinstall --no-cache-dir $HVD_WHEEL

echo "Pip install TensorBoard profiler plugin"
pip install tensorboard_plugin_profile tensorflow_addons tensorflow-datasets

cd $BASE_PATH
# KGF (2022-09-09):
MPICC="cc -shared -target-accel=nvidia80" pip install --force-reinstall --no-cache-dir --no-binary=mpi4py mpi4py

# KGF (2022-09-09): why did CUDA Aware mpi4py work for this install line, with PrgEnv-gnu but no craype-accel-nvidia80 module loaded, and no manual "CRAY_ACCEL_TARGET" exported ... but Horovod complains with:
# "MPIDI_CRAY_init: GPU_SUPPORT_ENABLED is requested, but GTL library is not linked"
# iff MPICH_GPU_SUPPORT_ENABLED=1 at runtime
#MPICC="cc -shared" pip install --force-reinstall --no-cache-dir --no-binary=mpi4py mpi4py

# echo Clone Mpi4py
# git clone $MPI4PY_REPO_URL
# cd mpi4py

# git checkout $MPI4PY_REPO_TAG

# LIBFAB_PATH=$(python -c "import os;x=os.environ['LD_LIBRARY_PATH'];x=x.split(':');x = [ i for i in x if 'libfabric' in i ];print(x[0])")
# echo $LD_LIBRARY_PATH
# echo $LIBFAB_PATH
# cat > mpi.cfg << EOF
# # MPI configuration for Polaris
# # ---------------------
# [mpi]

# mpi_dir              = $MPICH_DIR

# mpicc                = %(mpi_dir)s/bin/mpicc
# mpicxx               = %(mpi_dir)s/bin/mpicxx

# include_dirs         = %(mpi_dir)s/include
# library_dirs         = %(mpi_dir)s/lib

# ## extra_compile_args   =
# extra_link_args      = -L$LIBFAB_PATH -lfabric
# ## extra_objects        =

# EOF

# python setup.py build
# python setup.py bdist_wheel
# MPI4PY_WHL=$(find dist/ -name "mpi4py*.whl" -type f)
# mv $MPI4PY_WHL $WHEELS_PATH/
# MPI4PY_WHL=$(find $WHEELS_PATH/ -name "mpi4py*.whl" -type f)
# echo Install mpi4py $MPI4PY_WHL
# python -m pip install --force-reinstall $MPI4PY_WHL
echo "Pip install parallel h5py"
cd $BASE_PATH
git clone https://github.com/h5py/h5py.git
cd h5py
module load cray-hdf5-parallel
# corey used "CC=/opt/cray/pe/hdf5-parallel/1.12.1.3/bin/h5pcc"
# ❯ h5pcc -show
# cc -DpgiFortran -Wl,-rpath -Wl,/opt/cray/pe/hdf5-parallel/1.12.1.3/gnu/9.1/lib
export CC=cc
export HDF5_MPI="ON"
##export HDF5_DIR="/path/to/parallel/hdf5"  # If this isn't found by default
pip install .

echo "Pip install other packages"
pip install pandas matplotlib scikit-learn scipy pytest
pip install sacred wandb # Denis requests, April 2022


echo "Adding module snooper so we can tell what modules people are using"
# KGF: TODO, modify this path at the top of the script somehow; pick correct sitecustomize_polaris.py, etc.
# wont error out if first path does not exist; will just make a broken symbolic link
# https://github.com/argonne-lcf/PyModuleSnooper.git (fork of msalim2)
ln -s /soft/applications/PyModuleSnooper/sitecustomize.py $(python -c 'import site; print(site.getsitepackages()[0])')/sitecustomize.py

# DeepHyper stuff
# HARDCODE
pip install 'tensorflow_probability==0.24.0'
# KGF: 0.24.0 (2024-03-12) tested against TF 2.16.1 and JAX 0.4.25
# KGF: 0.21.0 (2023-08-04) tested against TF 2.13.x
# KGF: 0.20.0 (2023-05-08) tested against TF 2.12.x
# KGF: 0.19.0 (2022-12-06) tested against TF 2.11.x
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

    #####pip install ".[analytics,hvd,nas,autodeuq]"
    # Adding "sdv" optional requirement on Polaris with Python 3.8 force re-installed:
    # numpy-1.22.4, torch-1.13.1, which requires nvidia-cuda-nvrtc-cu11 + many other deps
    # No problem on ThetaGPU. Switching to Python 3.10 apparently avoids everything
    # TODO: if problems start again, test installing each of the sdv deps one-by-one (esp. ctgan)
    #####pip install ".[analytics,hvd,nas,hps-tl,autodeuq]"

    pip install ".[hps,hps-tl,nas,autodeuq,jax-gpu,automl,mpi,ray,redis-hiredis]"

    cd ..
    cd $BASE_PATH
else
    echo "Build DeepHyper tag $DH_REPO_TAG and Balsam from PyPI"
    ##pip install 'balsam-flow==0.3.8'  # balsam feature pinned to 0.3.8 from November 2019
    pip install "deephyper[analytics,hvd,nas,popt,autodeuq]==${DH_REPO_TAG}"  # otherwise, pulls 0.2.2 due to dependency conflicts?
fi

pip install 'libensemble'

#pip install 'libensemble[extras]'
# KGF: currently fails when building petsc4py wheel:
#          petsc: this package cannot be built as a wheel (???)
#     ...
#      error: PETSc not found
# No one is maintaining a PETSc module on Polaris right now; also don't really want to add a cross-module dep,
# nor build PETSc from source myself

# https://libensemble.readthedocs.io/en/main/introduction.html#dependencies
# - PETSc/TAO - Can optionally be installed by pip along with petsc4py (KGF: true??)
# PETSc and NLopt must be built with shared libraries enabled and be present in sys.path (e.g., via setting the PYTHONPATH environment variable).

# KGF: told to hold off on adding these as of 2022-12-20. Continue to advise user-local installs of fast-moving development branches
# pip install --pre balsam
# pip install parsl==1.3.0.dev0

# ---------------------------------------
# HARDCODE
# PyTorch Geometric Dependencies (2.3.x)
# torch 2.0.1 wheels just redirect to 2.0.0 wheels: https://data.pyg.org/whl/torch-2.0.1%2Bcu118.html

#pip install pyg_lib torch_sparse torch_cluster torch_scatter torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.1+cu${CUDA_VERSION_MAJOR}${CUDA_VERSION_MINOR}.html

# KGF: first 3x wheels still not working against our torch 2.0.1 built from source; torch_scatter, torch_spline_conv seem to work fine
# TODO: which git SHA are the pyg optional dep wheels built against? or is it an issue with the "torch.__version__" 2.0.0a0+gite9ebda2 (also in Pip)?

# KGF filed GitHub Issue about their wheels: https://github.com/pyg-team/pytorch_geometric/issues/8128
# HARDCODE
#pip install torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.1+cu${CUDA_VERSION_MAJOR}${CUDA_VERSION_MINOR}.html
pip install torch_spline_conv -f https://data.pyg.org/whl/torch-2.2.2+cu${CUDA_VERSION_MAJOR}1.html
# build the rest from source:
# KGF: note, the below LDFLAGS setting causes issues with "ldd libpyg.so" unable to find libpython3.10.so.1.0
# Need to "unset LDFLAGS" before the next line if installing it after this script in an interactive session
CUDAHOSTCXX=g++-12 CC=/usr/bin/gcc-12 CXX=/usr/bin/g++-12 pip install --verbose git+https://github.com/pyg-team/pyg-lib.git
# somehow uses CUDA.cmake from Pytorch repo, but doesnt respect USE_CUDNN, USE_CUSPARSELT settings
# next 2x require CPATH to be set?
export CPATH=${CUDA_TOOLKIT_BASE}/include:$CPATH
CUDAHOSTCXX=g++-12 CC=/usr/bin/gcc-12 CXX=/usr/bin/g++-12 pip install --verbose torch_sparse
#pip install --verbose torch_sparse
#pip install --verbose torch_scatter  # are we sure this needs to be built from source? used to install in above -f line
#pip install --verbose torch_cluster  # this takes a long time

# pyg-lib, torch-scatter, torch-sparse were required deps for pytorch_geometric 2.2.x and earlier, rest were optional. As of pytorch_geometric 2.3.x, the latter 2x pkgs were upstreamed to PyTorch. The 5x optional dependencies were kept around to offer minimal tweaks/use-cases: https://github.com/pyg-team/pytorch_geometric/releases/tag/2.3.0

# -------------------
# PyTorch Geometric Dependencies (2.2.x)
# --- For conda/2023-01-10: was hardcoding versions for dep binaries to PyTorch 1.13.1, CUDA 11.7 even though installing CUDA 11.8 above since CUDA 11.8 binaries werent provided until PyTorch Geometric with PyTorch 2.0. However, did not seem to work; import errors with pyg_lib, torch_scatter, torch_sparse pointing to compiled CUDA kernel minor version incompat. At a minimum, the latter 2x pkgs can be built from source against CUDA 11.8 (need to try pyg-lib)

#pip install pyg-lib torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-1.13.1+cu117.html

#pip install pyg-lib torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-1.13.1+cu${CUDA_VERSION_MAJOR}${CUDA_VERSION_MINOR}.html
# ---------------------------------------
pip install torch-geometric

# random inconsistencies that pop up with the specific "pip installs" from earlier
# HARDCODE
# KGF: retest
### pip install 'pytz>=2017.3' 'pillow>=6.2.0' 'django>=2.1.1'
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
# HARDCODE
git checkout v0.18.0
# KGF: this falls back to building a deprecated .egg format with easy_install, which puts an entry in
# mconda3/lib/python3.8/site-packages/easy-install.pth, causing read-only premissions problems in cloned
# environments.
###python setup.py install
# "We don't officially support building from source using pip, but if you do, you'll need to use the --no-build-isolation flag."

# KGF: build our own wheel, like in PyTorch and TF builds:
# HARDCODE
CUDAHOSTCXX=g++-12 CC=/usr/bin/gcc-12 CXX=/usr/bin/g++-12 python setup.py bdist_wheel
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
# HARDCODE
pip install 'onnx==1.16.0' 'onnxruntime-gpu==1.17.1'
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
# HARDCODE: warning, xformers will often pull in a newer version of torch wheel from PyPI, undoing source install
# and installing 11x nvidia-*-cu11 pkgs:
# nvidia-cublas-cu11 nvidia-cuda-cupti-cu11 nvidia-cuda-nvrtc-cu11 nvidia-cuda-runtime-cu11 nvidia-cudnn-cu11 nvidia-cufft-cu11 nvidia-curand-cu11 nvidia-cusolver-cu11 nvidia-cusparse-cu11 nvidia-nccl-cu11 nvidia-nvtx-cu11    # (but not nvidia-cuda-nvcc-cu11--- that comes from "jax[cuda11_pip]", which also overrides many of the above deps)
pip install --no-deps xformers
#pip install -U xformers   # requires PyTorch 2.0.1; at one point, it req 2.0.0 and it was optional but recommended to have triton 2.0.0 instead of 1.0.0
#pip install -v -U 'git+https://github.com/facebookresearch/xformers.git@main#egg=xformers'
pip install flash-attn --no-build-isolation
pip install scikit-image
pip install ipython   # KGF (2023-09-29): how was this missing from earlier scripts??
pip install line_profiler
pip install torch-tb-profiler
pip install torchinfo  # https://github.com/TylerYep/torchinfo successor to torchsummary (https://github.com/sksq96/pytorch-summary)
# https://docs.cupy.dev/en/stable/install.html
#pip install cupy-cuda${CUDA_VERSION_MAJOR}${CUDA_VERSION_MINOR}
# HARDCODE
pip install cupy-cuda${CUDA_VERSION_MAJOR}x
### pip install pycuda # KGF todo
# USE_SHIPPED_BOOST=True CUDAHOSTCXX=g++-12 CC=/usr/bin/gcc-12 CXX=/usr/bin/g++-12
pip install pytorch-lightning
pip install ml-collections
pip install gpytorch xgboost multiprocess py4j
# HARDCODE
CUDAHOSTCXX=g++-12 CC=/usr/bin/gcc-12 CXX=/usr/bin/g++-12 pip install --no-build-isolation git+https://github.com/FalkonML/falkon.git
pip install pykeops   # wants nonstandard env var set: CUDA_PATH=$CUDA_HOME
pip install hydra-core hydra_colorlog accelerate arviz pyright celerite seaborn xarray bokeh matplotx aim torchviz rich parse
pip install jupyter
pip install climetlab # Piyush Garg request

# Note, OpenAI's Triton, not NVIDIA's Triton: https://github.com/openai/triton/issues/156
# PyPI binary wheels 1.1.1, 1.0.0 might only work with CPython 3.6-3.9, not 3.10
#pip install "triton==1.0.0"
#pip install 'triton==2.0.0.dev20221202' || true

# But, DeepSpeed sparse support only supported triton v1.0 for a long time (KGF: still true as of September 2023)
# --- other features like stable diff. req/work with triton v2.1.0 as of late September 2023:
# https://github.com/microsoft/DeepSpeed/pull/4251  (triton>=2.0.0,<2.1.0, 2023-09-01)
# https://github.com/microsoft/DeepSpeed/pull/4278  (triton>=2.1.0, 2023-09-07)
# (initially, I thought you had to build DeepSpeed master > 0.10.3 from source for triton v2.1.0 support)

# ----------------
# HARDCODE
# cd $BASE_PATH
# echo "Install triton v2.1.0 from source"
# git clone https://github.com/openai/triton.git
# cd triton/python
# # git checkout v2.1.0
# ## did they change the tag numbering in summer 2023? No longer "v2.0", only "v2.0.0" (2023-03-02). Also no tag matching 2.1.0 wheel on PyPI
# v2.1.0 on 2023-09-01 (PyPI only) -- https://github.com/openai/triton/issues/2407
# v2.0.0 on 2023-03-02 (PyPI and GitHub)

# # Polaris-only issue:
# # CXX identified as CC or equiv. /opt/cray/pe/gcc/11.2.0/bin/g++ causes issues:
# # /opt/cray/pe/gcc/11.2.0/snos/include/g++/x86_64-suse-linux/bits/c++config.h:280:33: note: 'std::size_t' declared here
# # /soft/datascience/conda/2023-01-10/triton/include/triton/tools/graph.h:18:20: error: 'size_t' was not declared i n this scope; did you mean 'std::size_t'?
# CXX=/usr/bin/g++ pip install . -v
# # https://github.com/openai/triton/issues/808

# # this works for triton on ThetaGPU, even with Python 3.10
# ###pip install .

pip install triton  # v2.3.0 on PyPI on 2024-04-04
# recall, triton 2.1.0 is incompat with deepspeed 0.10.3 (2023-09-11), but works if building DS master from source
# after 2023-09-07
# https://github.com/microsoft/DeepSpeed/pull/4278
# https://github.com/microsoft/DeepSpeed/commit/e8ed7419ed40306100f0454bf85c6f4cc4d55f34

# -----------------
# https://www.deepspeed.ai/tutorials/ds4sci_evoformerattention/#31-installation
# DeepSpeed Evoformer requires CUTLASS, and looks for it at CUTLASS_PATH
# CUTLASS is a header-only template library and does not need to be built to be used by other projects.
# Client applications should target CUTLASS's include/ directory in their include paths.
cd $BASE_PATH
echo "Install CUTLASS from source"
git clone https://github.com/NVIDIA/cutlass
cd cutlass
# git checkout alcf_polaris  # Needed a hotfix on my fork+ special branch in Fall 2023
export CUTLASS_PATH="${BASE_PATH}/cutlass"
# https://github.com/NVIDIA/cutlass/blob/main/media/docs/quickstart.md
mkdir build && cd build
# strangely, https://github.com/NVIDIA/cutlass/blob/main/cuDNN.cmake doesnt respect $CUDNN_INCLUDE_DIR
export CUDNN_PATH=${CUDNN_BASE}
export CUDA_INSTALL_PATH=${CUDA_HOME}
export CUDACXX=${CUDA_INSTALL_PATH}/bin/nvcc
echo "About to run CMake for CUTLASS python = $(which python)"
conda info
CUDAHOSTCXX=g++-12 CC=/usr/bin/gcc-12 CXX=/usr/bin/g++-12 cmake .. -DCUTLASS_NVCC_ARCHS=80 -DCUTLASS_ENABLE_CUBLAS=ON -DCUTLASS_ENABLE_CUDNN=ON
# KGF: spurious errors with above CUTLASS cmake command in script (never encountered in interactive job)
# CMake Error at tools/library/CMakeLists.txt:285 (message):
#   Error generating library instances.  See
#   /soft/datascience/conda/2023-09-28/cutlass/build/tools/library/library_instance_generation.log
# ... (in that file:)
#
# Issue: CMake builds cutlass_library via "python setup_library.py develop --user"
# which puts it in:
# "PYTHONUSERBASE","/home/felker/.local/polaris/conda/2023-09-29" OR
# ~/.local/lib/python3.10/site-packages/easy-install.pth
# but then later in the CMake build chain, it cannot find/import it
# See https://github.com/felker/cutlass/commit/2368ed63d5dd2f4899873966c8b04912df6132fa

# KGF issue https://github.com/NVIDIA/cutlass/issues/1118
# echo "apply patch for CUTLASS #include <limits> in platform.h"
# cd ..
# git apply ~/cutlass_limits.patch
# cd build

make cutlass_profiler -j32  # fails at 98% cudnn_helpers.cpp.o without patch
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
git checkout v0.14.2
export CFLAGS="-I${CONDA_PREFIX}/include/"
export LDFLAGS="-L${CONDA_PREFIX}/lib/ -Wl,--enable-new-dtags,-rpath,${CONDA_PREFIX}/lib"
# KGF: above two lines need to be added to modulefile?

#DS_BUILD_SPARSE_ATTN=0 DS_BUILD_OPS=1 DS_BUILD_AIO=1 DS_BUILD_UTILS=1 bash install.sh --verbose

pip install deepspeed-kernels  # new since late 2023 for DeepSpeed-FastGen https://github.com/microsoft/DeepSpeed/pull/4604; also does a late upgrade of cmake 3.24 to 3.29

CUDAHOSTCXX=g++-12 CC=/usr/bin/gcc-12 CXX=/usr/bin/g++-12 NVCC_PREPEND_FLAGS="--forward-unknown-opts" DS_BUILD_SPARSE_ATTN=0 DS_BUILD_OPS=1 DS_BUILD_AIO=1 pip install --verbose . ### --global-option="build_ext" --global-option="-j16"
# the parallel build options seem to cause issues

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
# Apex (for Megatron-Deepspeed)
CUDAHOSTCXX=g++-12 CC=/usr/bin/gcc-12 CXX=/usr/bin/g++-12 python3 -m pip install \
	-vv \
	--disable-pip-version-check \
	--no-cache-dir \
	--no-build-isolation \
	--config-settings "--build-option=--cpp_ext" \
	--config-settings "--build-option=--cuda_ext" \
	"git+https://github.com/NVIDIA/apex.git@24.04.01"  # April 27 2024 release; still shows up as apex-0.1
#       "git+https://github.com/NVIDIA/apex.git@52e18c894223800cb611682dce27d88050edf1de"
# commit corresponds to PR from Sept 2023: https://github.com/NVIDIA/apex/pull/1721

# KGF TODO: myriad issues with Megatron-LM and forked Megatron-DeepSpeed packaging:
# - everything assumes that it was installed as --editable, everyone has write access, etc.
# - detecting old NCCL version 2.14.3+cuda11.8
# - model/vision/* not importable (fixed on ALCF fork)
# - IBV_FORK_SAFE=1 still required?? leads to 100x slowdown on Polaris

# https://github.com/microsoft/Megatron-DeepSpeed/pull/173#issuecomment-1755839097
python3 -m pip install "git+https://github.com/microsoft/Megatron-DeepSpeed.git"

# HARDCODE
###pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
# --- Building wheels for collected packages: nvidia-cublas-cu11, nvidia-cuda-cupti-cu11, nvidia-cuda-nvcc-cu11, nvidia-cuda-runtime-cu11, nvidia-cudnn-cu11

# KGF POTENTIAL ISSUE SEPTEMBER 2023:
# Try building from source, OR getting wheel that links to local CUDA and cuDNN
# See https://jax.readthedocs.io/en/latest/developer.html#building-from-source
# OR
pip install --upgrade "jax[cuda${CUDA_VERSION_MAJOR}_local]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
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
# KGF: still need to apply manual postfix for the 4x following warnings that appear whenever "conda list" or other commands are run
# WARNING conda.gateways.disk.delete:unlink_or_rename_to_trash ... /lus/theta-fs0/software/thetagpu/conda/deephyper/0.2.5/mconda3/conda-meta/setuptools-52.0.0-py38h06a4308_0.json

# KGF: Do "chmod -R u+w ." in mconda3/conda-meta/, run "conda list", then "chmod -R a-w ."


# https://github.com/deephyper/deephyper/issues/110
# KGF: check that CONDA_DIR/mconda3/lib/python3.8/site-packages/easy-install.pth does not exist as an empty file
# rm it to prevent it from appearing in cloned conda environments (with read-only permissions), preventing users
# from instalilng editable pip installs in their own envs!
