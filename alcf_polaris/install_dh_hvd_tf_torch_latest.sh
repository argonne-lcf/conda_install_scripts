#!/bin/bash

# As of May 2022
# This script will install TensorFlow, PyTorch, and Horovod on Polaris
# 1 - Login to Polaris login-node
# 2 - Run 'bash <this script> /path/to/install/base/'
# 3 - script installs everything down in $PWD/deephyper/...
# 4 - wait for it to complete

BASE_PATH=$1

echo Installing into $BASE_PATH
read -p "Are you sure? " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]
then
    echo Installing
else
   exit -1
fi

cd $BASE_PATH

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

# Using our own nvidia environment so swap to GNU env
module switch PrgEnv-nvidia PrgEnv-gnu


# unset *_TAG variables to build latest master
#DH_REPO_TAG="0.2.5"
#DH_REPO_URL=https://github.com/deephyper/deephyper.git

#TF_REPO_TAG="e5a6d2331b11e0e5e4b63a0d7257333ac8b8262a" # requires NumPy 1.19.x
#TF_REPO_TAG="v2.7.0"
#PT_REPO_TAG="v1.10.0"
HOROVOD_REPO_TAG="v0.24.3" # v0.22.1 released on 2021-06-10 should be compatible with TF 2.6.x and 2.5.x
TF_REPO_URL=https://github.com/tensorflow/tensorflow.git
HOROVOD_REPO_URL=https://github.com/uber/horovod.git
PT_REPO_URL=https://github.com/pytorch/pytorch.git


###########################################
# CUDA path and version information
############################

CUDA_DEPS_BASE=/home/parton/cuda

CUDA_VERSION_MAJOR=11
CUDA_VERSION_MINOR=4
CUDA_VERSION_MINI=4
CUDA_VERSION_BUILD=470.82.01
CUDA_VERSION=$CUDA_VERSION_MAJOR.$CUDA_VERSION_MINOR
CUDA_VERSION_FULL=$CUDA_VERSION.$CUDA_VERSION_MINI
CUDA_BASE=$CUDA_DEPS_BASE/cuda_${CUDA_VERSION_FULL}_${CUDA_VERSION_BUILD}_linux

CUDNN_VERSION_MAJOR=8
CUDNN_VERSION_MINOR=2
CUDNN_VERSION_EXTRA=4.15
CUDNN_VERSION=$CUDNN_VERSION_MAJOR.$CUDNN_VERSION_MINOR.$CUDNN_VERSION_EXTRA
CUDNN_BASE=$CUDA_DEPS_BASE/cudnn-$CUDA_VERSION-linux-x64-v$CUDNN_VERSION

NCCL_VERSION_MAJOR=2
NCCL_VERSION_MINOR=12.10-1
NCCL_VERSION=$NCCL_VERSION_MAJOR.$NCCL_VERSION_MINOR
NCCL_BASE=$CUDA_DEPS_BASE/nccl_$NCCL_VERSION+cuda${CUDA_VERSION}_x86_64
# KGF: no Extended Compatibility in NCCL --- use older NCCL version built with CUDA 11.0 until
# GPU device kernel driver upgraded from 11.0 ---> 11.4 in November 2021
#NCCL_BASE=$CUDA_DEPS_BASE/nccl_2.9.9-1+cuda11.0_x86_64


#############################################
## INSTALLING MiniConda
###########

# set Conda installation folder and where downloaded content will stay
CONDA_PREFIX_PATH=$BASE_PATH/mconda3
DOWNLOAD_PATH=$BASE_PATH/DOWNLOADS
WHEELS_PATH=$BASE_PATH/wheels

mkdir -p $CONDA_PREFIX_PATH
mkdir -p $DOWNLOAD_PATH
mkdir -p $WHEELS_PATH

# Download and install conda for a base python installation
CONDAVER='py38_4.10.3'
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

#########
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
   export https_proxy=http://proxy.alcf.anl.gov:3128
   export http_proxy=http://proxy.alcf.anl.gov:3128
fi

module switch PrgEnv-nvidia PrgEnv-gnu

export CUDA_BASE=$CUDA_BASE
export CUDNN_BASE=$CUDNN_BASE
export NCCL_BASE=$NCCL_BASE
export LD_LIBRARY_PATH=\$CUDA_BASE/lib64:\$CUDNN_BASE/lib64:\$NCCL_BASE/lib:\$LD_LIBRARY_PATH
export PATH=\$CUDA_BASE/bin:\$PATH
EOF

#######
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
env_prompt: "(${BASE_PATH}/{default_env}) "
pkgs_dirs:
   - \$HOME/.conda/pkgs
EOF

# move to base install directory
cd $BASE_PATH

# setup conda environment
source $CONDA_PREFIX_PATH/setup.sh

# install dependencies/tools from conda
conda install -y cmake

# KGF: probably dont need a third (removed) network check--- proxy env vars inherited from either sourced setup.sh
# and/or first network check. Make sure "set+e" during above sourced setup.sh since the network check "wget" might
# return nonzero code if network is offline

echo CONDA BINARY: $(which conda)
echo CONDA VERSION: $(conda --version)
echo PYTHON VERSION: $(python --version)

set -e

################################################
### Install TensorFlow
########

pip install tensorflow

#################################################
### Install PyTorch
########

pip install torch

################################################
### Install Horovod
########

cd $BASE_PATH

echo Clone Horovod

git clone --recursive $HOROVOD_REPO_URL
cd horovod

if [[ -z "$HOROVOD_REPO_TAG" ]]; then
    echo Checkout Horovod master
else
    echo Checkout Horovod tag $HOROVOD_REPO_TAG
    git checkout --recurse-submodules $HOROVOD_REPO_TAG
fi

echo Build Horovod Wheel using MPI from $MPI and NCCL from ${NCCL_BASE}
# https://horovod.readthedocs.io/en/stable/gpus_include.html
# If you installed NCCL 2 using the nccl-<version>.txz package, you should specify the path to NCCL 2 using the HOROVOD_NCCL_HOME environment variable.
# add the library path to LD_LIBRARY_PATH environment variable or register it in /etc/ld.so.conf.
#export LD_LIBRARY_PATH=$CRAY_MPICH_PREFIX/lib-abi-mpich:$NCCL_BASE/lib:$LD_LIBRARY_PATH
#export PATH=$CRAY_MPICH_PREFIX/bin:$PATH

HOROVOD_WITH_MPI=1 python setup.py bdist_wheel
HVD_WHL=$(find dist/ -name "horovod*.whl" -type f)
cp $HVD_WHL $WHEELS_PATH/
HVD_WHEEL=$(find $WHEELS_PATH/ -name "horovod*.whl" -type f)
echo Install Horovod $HVD_WHEEL
pip install --force-reinstall $HVD_WHEEL

echo Pip install TensorBoard profiler plugin
pip install tensorboard_plugin_profile tensorflow_addons
echo Pip install other packages
pip install pandas h5py matplotlib scikit-learn scipy pytest
pip install sacred wandb # Denis requests, April 2022

exit 0

echo Adding module snooper so we can tell what modules people are using
ln -s /lus/theta-fs0/software/datascience/PyModuleSnooper/sitecustomize.py $(python -c 'import site; print(site.getsitepackages()[0])')/sitecustomize.py

# DeepHyper stuff
#export PATH=$MPI/bin:$PATH  # hvd optional feature will build mpi4py wheel

pip install 'tensorflow_probability==0.14.0'
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
    pip install 'balsam-flow==0.3.8'  # balsam feature pinned to 0.3.8 from November 2019
    pip install "deephyper[analytics,balsam,deepspace]==${DH_REPO_TAG}"  # otherwise, pulls 0.2.2 due to dependency conflicts?
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
