#!/bin/bash

# get the folder where this script is living
THISDIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd -LP )
PT_INSTALL_BASE_DIR=$THISDIR/pt_master/2021-05-12

# As of Nov 23 2020
# This script will install pytorch and horovod from scratch on ThetaGPU
# 1 - Grab worker node interactively for 120 min
# 2 - Run 'bash install_pt.sh'
# 3 - script installs everything down in $PWD/pt-install
# 4 - wait for it to complete

# PyTorch source and version
PT_REPO_URL=https://github.com/pytorch/pytorch.git
PT_REPO_TAG=master

# Horovod source and version
HOROVOD_REPO_URL=https://github.com/uber/horovod.git
HOROVOD_REPO_TAG="v0.21.3"

# MPI source on ThetaGPU
MPI=/lus/theta-fs0/software/thetagpu/openmpi-4.0.5

CUDA_DEPS_BASE=/lus/theta-fs0/software/thetagpu/cuda

# CUDA path and version information
CUDA_VERSION_MAJOR=11
CUDA_VERSION_MINOR=0
CUDA_VERSION=$CUDA_VERSION_MAJOR.$CUDA_VERSION_MINOR
CUDA_BASE=/usr/local/cuda-$CUDA_VERSION

CUDNN_VERSION_MAJOR=8
CUDNN_VERSION_MINOR=1
CUDNN_VERSION_EXTRA=1.33
CUDNN_VERSION=$CUDNN_VERSION_MAJOR.$CUDNN_VERSION_MINOR.$CUDNN_VERSION_EXTRA
CUDNN_BASE=$CUDA_DEPS_BASE/cudnn-$CUDA_VERSION-linux-x64-v$CUDNN_VERSION

NCCL_VERSION_MAJOR=2
NCCL_VERSION_MINOR=9.6-1
NCCL_VERSION=$NCCL_VERSION_MAJOR.$NCCL_VERSION_MINOR
NCCL_BASE=$CUDA_DEPS_BASE/nccl_$NCCL_VERSION+cuda${CUDA_VERSION}_x86_64

TENSORRT_VERSION_MAJOR=7
TENSORRT_VERSION_MINOR=2.2.3
TENSORRT_VERSION=$TENSORRT_VERSION_MAJOR.$TENSORRT_VERSION_MINOR
TENSORRT_BASE=$CUDA_DEPS_BASE/TensorRT-$TENSORRT_VERSION.Ubuntu-18.04.x86_64-gnu.cuda-$CUDA_VERSION.cudnn$CUDNN_VERSION_MAJOR.$CUDNN_VERSION_MINOR



# set install path
WHEEL_DIR=$PT_INSTALL_BASE_DIR/wheels
mkdir -p $WHEEL_DIR

# confirm install path
echo Installing pytorch into $PT_INSTALL_BASE_DIR
read -p "Are you sure? " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]
then
    echo OK, you asked for it...
else
   exit -1
fi

set -e

# needed for outside communication on ThetaGPU
export https_proxy=http://proxy.tmi.alcf.anl.gov:3128
export http_proxy=http://proxy.tmi.alcf.anl.gov:3128

# set Conda installation folder and where downloaded content will stay
CONDA_PREFIX_PATH=$PT_INSTALL_BASE_DIR/mconda3
DOWNLOAD_PATH=$PT_INSTALL_BASE_DIR/DOWNLOADS

mkdir -p $CONDA_PREFIX_PATH
mkdir -p $DOWNLOAD_PATH

# Download and install conda for a base python installation
CONDAVER=latest
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
preferred_shell=$(basename $SHELL)

if [ -n "\$ZSH_EVAL_CONTEXT" ]; then
    DIR=\$( cd "\$( dirname "\$0" )" && pwd )
else  # bash, sh, etc.
    DIR=\$( cd "\$( dirname "\${BASH_SOURCE[0]}" )" && pwd )
fi

eval "\$(\$DIR/bin/conda shell.\${preferred_shell} hook)"

export https_proxy=http://proxy.tmi.alcf.anl.gov:3128
export http_proxy=http://proxy.tmi.alcf.anl.gov:3128

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

cat > .condarc << EOF
env_prompt: "(\$ENV_NAME/\$CONDA_DEFAULT_ENV) "
pkgs_dirs:
   - \$HOME/.conda/pkgs
EOF

# move to base install directory
cd $PT_INSTALL_BASE_DIR

# setup conda environment
source $CONDA_PREFIX_PATH/setup.sh

# re-add proxys
export https_proxy=http://proxy.tmi.alcf.anl.gov:3128
export http_proxy=http://proxy.tmi.alcf.anl.gov:3128

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

echo Conda install some dependencies
conda install -y cmake zip unzip numpy ninja pyyaml mkl mkl-include setuptools cmake cffi typing_extensions future six requests dataclasses
conda update -y pip
echo Clone Tensorflow
cd $PT_INSTALL_BASE_DIR
git clone --recursive $PT_REPO_URL
cd pytorch
echo Checkout Tensorflow tag $PT_REPO_TAG
git checkout $PT_REPO_TAG

echo Install PyTorch

export CUDA_TOOLKIT_ROOT_DIR=$CUDA_BASE
export NCCL_ROOT_DIR=$NCCL_BASE
export CUDNN_ROOT=$CUDNN_BASE
export USE_TENSORRT=ON
export TENSORRT_ROOT=$TENSORRT_BASE
export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
export TENSORRT_LIBRARY=$TENSORRT_BASE/lib/libmyelin.so
export TENSORRT_LIBRARY_INFER=$TENSORRT_BASE/lib/libnvinfer.so
export TENSORRT_LIBRARY_INFER_PLUGIN=$TENSORRT_BASE/lib/libnvinfer_plugin.so
export TENSORRT_INCLUDE_DIR=$TENSORRT_BASE/include
python setup.py bdist_wheel
PT_WHEEL=$(find dist/ -name "torch*.whl" -type f)
echo copying pytorch wheel file $PT_WHEEL
cp $PT_WHEEL $WHEEL_DIR/
cd $WHEEL_DIR
echo pip installing $(basename $PT_WHEEL)
pip install $(basename $PT_WHEEL)

cd $PT_INSTALL_BASE_DIR

echo Clone Horovod $HOROVOD_REPO_TAG git repo

git clone --recursive $HOROVOD_REPO_URL
cd horovod
git checkout $HOROVOD_REPO_TAG

echo Build Horovod using MPI from $MPI
export LD_LIBRARY_PATH=$MPI/lib:$LD_LIBRARY_PATH
export PATH=$MPI/bin:$PATH

HOROVOD_NCCL_HOME=$NCCL_BASE HOROVOD_CMAKE=$(which cmake) HOROVOD_GPU_OPERATIONS=NCCL HOROVOD_WITHOUT_TENSORFLOW=1 HOROVOD_WITH_PYTORCH=1 HOROVOD_WITHOUT_MXNET=1 python setup.py bdist_wheel
HVD_WHL=$(find dist/ -name "horovod*.whl" -type f)
echo copying horovod wheel $HVD_WHL
cp $HVD_WHL $WHEEL_DIR/
cd $WHEEL_DIR
echo pip installing $(basename $HVD_WHL)
pip install $(basename $HVD_WHL)

echo Cleaning up
rm -rf $DOWNLOAD_PATH

echo install extras
pip install --no-deps torchvision pillow
conda install -y pandas matplotlib scikit-learn scipy h5py

echo Adding module snooper so we can tell what modules people are using
ln -s /lus/theta-fs0/software/datascience/PyModuleSnooper/sitecustomize.py $(python -c 'import site; print(site.getsitepackages()[0])')/sitecustomize.py

echo Cleaning up
chmod -R u+w $DOWNLOAD_PATH/
rm -rf $DOWNLOAD_PATH
echo Change to read-only
chmod -R a-w $PT_INSTALL_BASE_DIR/
