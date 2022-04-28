#!/bin/bash

set -e

PYTHON_VERSION=3
CONDA_VERSION='py38_4.10.3'
#BASE_DIR=$PWD/miniconda$PYTHON_VERSION

# if [ $# -eq 0 ]
# then
#    PREFIX_PATH=$BASE_DIR/$CONDA_VERSION
# else
#    PREFIX_PATH=$1
# fi

PREFIX_PATH=$PWD/2021-09-22-smartsim

WHEEL_DIR=$PREFIX_PATH/wheels
DOWNLOAD_PATH=$PREFIX_PATH/DOWNLOADS
CONDA_INST_PATH=$PREFIX_PATH/mconda3


# unset *_TAG variables to build latest master
#DH_REPO_TAG="0.2.5"
DH_REPO_URL=https://github.com/deephyper/deephyper.git

# Horovod source and version
HOROVOD_REPO_URL=https://github.com/uber/horovod.git
HOROVOD_REPO_TAG=v0.22.1

echo Installing into $PREFIX_PATH
read -p "Are you sure? " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]
then
    echo Installing
else
   exit -1
fi

# change umask to remove group write
umask 0022

mkdir -p $PREFIX_PATH
mkdir -p $DOWNLOAD_PATH
mkdir -p $WHEEL_DIR

MINICONDA_INSTALL_FILE=Miniconda$PYTHON_VERSION-$CONDA_VERSION-Linux-x86_64.sh

if [[ ! -f $DOWNLOAD_PATH/$MINICONDA_INSTALL_FILE ]] || [[ `find $DOWNLOAD_PATH/$MINICONDA_INSTALL_FILE -ctime +30` ]]; then
   rm -f $DOWNLOAD_PATH/$MINICONDA_INSTALL_FILE
   echo Downloading miniconda installer
   wget https://repo.continuum.io/miniconda/$MINICONDA_INSTALL_FILE -P $DOWNLOAD_PATH

   chmod +x $DOWNLOAD_PATH/Miniconda$PYTHON_VERSION-$CONDA_VERSION-Linux-x86_64.sh
fi

echo Installing Miniconda
$DOWNLOAD_PATH/Miniconda$PYTHON_VERSION-$CONDA_VERSION-Linux-x86_64.sh -b -p $CONDA_INST_PATH  -u

echo moving into $CONDA_INST_PATH
cd $CONDA_INST_PATH


PYTHON_VER=$(ls -d lib/python?.? | tail -c4)
echo PYTHON_VER=$PYTHON_VER

# create a setup file
cat > setup.sh << EOF
preferred_shell=\$(basename \$SHELL)

if [ -n "\$ZSH_EVAL_CONTEXT" ]; then
    DIR=\$( cd "\$( dirname "\$0" )" && pwd )
else  # bash, sh, etc.
    DIR=\$( cd "\$( dirname "\${BASH_SOURCE[0]}" )" && pwd )
fi

eval "\$(\$DIR/bin/conda shell.\${preferred_shell} hook)"
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

cat > modulefile << EOF
#%Module2.0
## miniconda$PYTHON_VERSION modulefile
##
proc ModulesHelp { } {
   global CONDA_LEVEL PYTHON_LEVEL MINICONDA_LEVEL
   puts stderr "This module will add Miniconda \$MINICONDA_LEVEL to your environment with conda version \$CONDA_LEVEL and python version \$PYTHON_LEVEL"
}

set _module_name  [module-info name]
set is_module_rm  [module-info mode remove]
set sys           [uname sysname]
set os            [uname release]
set HOME          $::env(HOME)

set PYTHON_LEVEL                 $PYTHON_VER
set CONDA_LEVEL                  $CONDA_VERSION
set MINICONDA_LEVEL              $PYTHON_VERSION
set CONDA_PREFIX                 $CONDA_INST_PATH
setenv CONDA_PREFIX              \$CONDA_PREFIX
setenv PYTHONUSERBASE            \$HOME/.local/\$_module_name
setenv ENV_NAME                  \$_module_name
setenv PYTHONSTARTUP             \$CONDA_PREFIX/etc/pythonstart
puts stdout "source \$CONDA_PREFIX/setup.sh"

module-whatis  "miniconda installation"
EOF

cat > .condarc << EOF
env_prompt: "(\$ENV_NAME/\$CONDA_DEFAULT_ENV) "
pkgs_dirs:
   - \$HOME/.local/conda/\$ENV_NAME/pkgs
   - \$CONDA_PREFIX/pkgs
EOF

# setup area
echo setting up conda environment
module load $(pwd)/modulefile
conda config --remove channels intel || true
conda install -y cmake
conda install -y -c conda-forge mamba
conda install -y -c conda-forge git-lfs
git lfs install

echo CONDA BINARY: $(which conda)
echo CONDA VERSION: $(conda --version)
pip install --upgrade pip
echo PIP VERSION: $(pip --version)

echo Pip installing TensorFlow and TF probability

# install tensorflow depenedencies
pip install 'tensorflow==2.6.0'

pip install 'tensorflow_probability==0.14.0'
# KGF: 0.14.0 (2021-09-15) only compatible with TF 2.6.0
# KGF: 0.13.0 (2021-06-18) only compatible with TF 2.5.0

# install pytorch
echo Pip installing PyTorch
pip install 'torch==1.7.1'

echo Pip installing other tools
pip install scikit-learn scikit-image pandas matplotlib h5py scikit-optimize virtualenv tensorboard_plugin_profile tensorflow_addons scipy

echo install smartsim
git clone https://github.com/CrayLabs/SmartSim.git --depth=1 --branch v0.3.2 smartsim-0.3.2
cd smartsim-0.3.2
pip install -e .[dev,ml]
smart -v --device cpu

echo install smartredis
cd ..
git clone https://github.com/CrayLabs/SmartRedis.git --depth=1 --branch v0.2.0 smartredis-0.2.0
cd smartredis-0.2.0
pip install -e .[dev]

export CC=/opt/gcc/9.3.0/bin/gcc
export CXX=/opt/gcc/9.3.0/bin/g++
make deps
make test-deps
make lib

########
### Install Horovod
########
cd $PREFIX_PATH
module swap PrgEnv-intel PrgEnv-gnu
module swap gcc gcc/8.3.0
export CRAY_CPU_TARGET=mic-knl
echo Clone Horovod $HOROVOD_REPO_TAG git repo
echo $CC $CXX
git clone --recursive $HOROVOD_REPO_URL
cd horovod
git checkout $HOROVOD_REPO_TAG

#echo Build Horovod Wheel using MPI from $MPI
#export LD_LIBRARY_PATH=$MPI/lib:$LD_LIBRARY_PATH
#export PATH=$MPI/bin:$PATH

HOROVOD_CMAKE=$(which cmake) HOROVOD_WITH_TENSORFLOW=1 HOROVOD_WITH_PYTORCH=1 HOROVOD_WITHOUT_MXNET=1 CC=$(which cc) CXX=$(which CC) python setup.py bdist_wheel
HVD_WHL=$(find dist/ -name "horovod*.whl" -type f)
cp $HVD_WHL $WHEEL_DIR/
HVD_WHEEL=$(find $WHEEL_DIR/ -name "horovod*.whl" -type f)
echo Install Horovod $HVD_WHEEL
pip install --force-reinstall $HVD_WHEEL

# KGF: sometimes throws errors?
set +e
pip install mpi4py
set -e

pip install 'tensorflow_probability==0.14.0'
pip install 'deephyper==0.3.0'

pip install "pillow!=8.3.0,>=6.2.0"
pip install --no-deps torchvision



chmod -R a-w $PREFIX_PATH

set +e
