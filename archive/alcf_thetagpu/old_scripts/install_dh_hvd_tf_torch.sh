#!/bin/bash

# As of June 14 2021
# This script will install DeepHyper, TensorFlow, PyTorch, and Horovod on ThetaGPU
# 1 - Copy .whl files from tf_2/**/wheels/ and pt_master/**/wheels/ to deephyper/VER/wheels/
# 2 - Run 'bash install_tensorflow.sh'
# 3 - script installs everything down in $PWD/deephyper
# 4 - wait for it to complete

DH_REPO_TAG="0.2.5"

# unused; for reference
TF_REPO_TAG="v2.4.1"
TORCH_TAG="1.9.0a0+gitff982ef"
HOROVOD_REPO_TAG="v0.21.3"
TF_REPO_URL=https://github.com/tensorflow/tensorflow.git
DH_REPO_URL=https://github.com/deephyper/deephyper
HOROVOD_REPO_URL=https://github.com/uber/horovod.git

# where to install relative to current path
DH_INSTALL_SUBDIR=deephyper/${DH_REPO_TAG}

# MPI source on ThetaGPU
MPI=/lus/theta-fs0/software/thetagpu/openmpi-4.0.5

# CUDA path and version information
CUDA_VERSION_MAJOR=11
CUDA_VERSION_MINOR=0
CUDA_VERSION=$CUDA_VERSION_MAJOR.$CUDA_VERSION_MINOR
CUDA_BASE=/usr/local/cuda-$CUDA_VERSION

CUDA_DEPS_BASE=/lus/theta-fs0/software/thetagpu/cuda

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
TENSORRT_VERSION_MINOR=2.3.4
TENSORRT_VERSION=$TENSORRT_VERSION_MAJOR.$TENSORRT_VERSION_MINOR
TENSORRT_BASE=$CUDA_DEPS_BASE/TensorRT-$TENSORRT_VERSION.Ubuntu-18.04.x86_64-gnu.cuda-$CUDA_VERSION.cudnn$CUDNN_VERSION_MAJOR.$CUDNN_VERSION_MINOR

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
echo Installing tensorflow into $DH_INSTALL_BASE_DIR
#read -p "Are you sure? " -n 1 -r
#echo
#if [[ $REPLY =~ ^[Yy]$ ]]
#then
#    echo OK, you asked for it...
#else
#   exit -1
#fi


# needed for outside communication on ThetaGPU
wget -q --spider http://google.com
if [ $? -eq 0 ]; then
    echo "Network Online"
else
   echo "Network Offline, setting proxy envs"
   export https_proxy=http://proxy.tmi.alcf.anl.gov:3128
   export http_proxy=http://proxy.tmi.alcf.anl.gov:3128
fi

set -e

# set Conda installation folder and where downloaded content will stay
CONDA_PREFIX_PATH=$DH_INSTALL_BASE_DIR/mconda3
DOWNLOAD_PATH=$DH_INSTALL_BASE_DIR/DOWNLOADS

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

# test network
wget -q --spider http://google.com
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

cat > .condarc << EOF
env_prompt: "(\$ENV_NAME/\$CONDA_DEFAULT_ENV) "
pkgs_dirs:
   - \$HOME/.conda/pkgs
EOF


# move to base install directory
cd $DH_INSTALL_BASE_DIR

# setup conda environment
source $CONDA_PREFIX_PATH/setup.sh

set +e
# needed for outside communication on ThetaGPU
wget -q --spider http://google.com
if [ $? -eq 0 ]; then
    echo "Network Online"
else
   echo "Network Offline, setting proxy envs"
   export https_proxy=http://proxy.tmi.alcf.anl.gov:3128
   export http_proxy=http://proxy.tmi.alcf.anl.gov:3128
fi

set -e


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
conda install -y cmake zip unzip
conda update -y pip

cd $DH_INSTALL_BASE_DIR

# KGF: random deps that pip mildlly complained about
pip install toml
pip install 'appdirs<2,>=1.4.3'
pip install 'pyyaml>=5.1'
pip install 'filelock<4,>=3.0.0'
pip install 'typing-extensions~=3.7.4'

echo Install Tensorflow Dependencies
pip install -U pip six 'numpy<1.19.0' wheel setuptools mock 'future>=0.17.1' 'gast==0.3.3' typing_extensions portpicker
pip install -U keras_applications --no-deps
pip install -U keras_preprocessing --no-deps

TF_WHEEL=$(find $WHEEL_DIR/ -name "tensorflow*.whl" -type f)
echo Install TensorFlow $TF_WHEEL
pip install --force-reinstall $TF_WHEEL

PT_WHEEL=$(find $WHEEL_DIR/ -name "torch*.whl" -type f)
echo Install PyTorch $PT_WHEEL
pip install --force-reinstall $PT_WHEEL

HVD_WHEEL=$(find $WHEEL_DIR/ -name "horovod*.whl" -type f)
echo Install Horovod $HVD_WHEEL
pip install --force-reinstall $HVD_WHEEL

echo Install Tensorboard profiler plugin
pip install tensorboard_plugin_profile tensorflow_addons
echo Install other packages
pip install pandas h5py matplotlib scikit-learn scipy

echo Adding module snooper so we can tell what modules people are using
ln -s /lus/theta-fs0/software/datascience/PyModuleSnooper/sitecustomize.py $(python -c 'import site; print(site.getsitepackages()[0])')/sitecustomize.py

# DeepHyper stuff
pip install 'balsam-flow==0.3.8'  # balsam feature pinned to 0.3.8 from November 2019
export PATH=$MPI/bin:$PATH  # hvd optional feature will build mpi4py wheel
pip install "deephyper[analytics,balsam]==${DH_REPO_TAG}"  # otherwise, pulls 0.2.2 due to dependency conflicts?
#pip install 'deephyper[analytics,balsam,hvd]==0.2.5'  # otherwise, pulls 0.2.2 due to dependency conflicts?

pip install deepspace  # KGF: not installed by default in above command?

# git clone https://github.com/deephyper/deephyper.git
# cd deephyper/
# pip install -e '.[analytics,balsam,hvd]'

# KGF: avoid 4x of the following warnings whenever "conda list" or other commands are run
# WARNING conda.gateways.disk.delete:unlink_or_rename_to_trash ... /lus/theta-fs0/software/thetagpu/conda/deephyper/0.2.5/mconda3/conda-meta/setuptools-52.0.0-py38h06a4308_0.json
#conda clean --all -y

# KGF: didnt fix it. Had to "chmod -R u+w ." in mconda3/conda-meta/, run "conda list", then "chmod -R a-w ."


echo Cleaning up
chmod -R u+w $DOWNLOAD_PATH/
rm -rf $DOWNLOAD_PATH

chmod -R a-w $DH_INSTALL_BASE_DIR/

# KGF: how to install /remove/update/etc packages in this module's Conda base env after the script completes?
# "source /lus/theta-fs0/software/thetagpu/conda/deephyper/0.2.5/mconda3/setup.sh" seems insufficient, even after changing permissions

# And pip install often tries to install in /home/felker/.local/lib/python3.8/site-packages/
# See "python -m site". Can I change ENABLE_USER_SITE: False?
# Need to write to:
# /lus/theta-fs0/software/thetagpu/conda/deephyper/0.2.5/mconda3/lib/python3.8/site-packages/deephyper


# KGF: the conda environment is not auto-activated when loading the module?
