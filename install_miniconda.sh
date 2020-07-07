#!/bin/bash

PYTHON_VERSION=3
CONDA_VERSION=latest
BASE_DIR=$PWD/miniconda$PYTHON_VERSION
DOWNLOAD_PATH=$BASE_DIR/DOWNLOADS

if [ $# -eq 0 ]
then
   PREFIX_PATH=$BASE_DIR/$CONDA_VERSION
else
   PREFIX_PATH=$1
fi 

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

MINICONDA_INSTALL_FILE=Miniconda$PYTHON_VERSION-$CONDA_VERSION-Linux-x86_64.sh

if [ ! -f $DOWNLOAD_PATH/$MINICONDA_INSTALL_FILE ]; then
   echo Downloading miniconda installer
   wget https://repo.continuum.io/miniconda/$MINICONDA_INSTALL_FILE -P $DOWNLOAD_PATH
   
   chmod +x $DOWNLOAD_PATH/Miniconda$PYTHON_VERSION-$CONDA_VERSION-Linux-x86_64.sh
fi

echo Installing Miniconda
$DOWNLOAD_PATH/Miniconda$PYTHON_VERSION-$CONDA_VERSION-Linux-x86_64.sh -b -p $PREFIX_PATH -u

echo moving into $PREFIX_PATH
cd $PREFIX_PATH


PYTHON_VER=$(ls -d lib/python?.? | tail -c4)
echo PYTHON_VER=$PYTHON_VER

# create a setup file
cat > setup.sh << EOF
DIR=\$( cd "\$( dirname "\${BASH_SOURCE[0]}" )" && pwd )
eval "\$(\$DIR/bin/conda shell.bash hook)"
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

set PYTHON_LEVEL                 $PYTHON_VER
set CONDA_LEVEL                  $CONDA_VERSION
set MINICONDA_LEVEL              $PYTHON_VERSION
set CONDA_PREFIX                 $PREFIX_PATH
setenv CONDA_PREFIX              \$CONDA_PREFIX
setenv ENV_NAME                  \$_module_name
setenv PYTHONSTARTUP             \$CONDA_PREFIX/etc/pythonstart
puts stdout "source \$CONDA_PREFIX/setup.sh"

module-whatis  "miniconda installation"
EOF

cat > .condarc << EOF
env_prompt: "(\$ENV_NAME/\$CONDA_DEFAULT_ENV) "
pkgs_dirs:
   - \$CONDA_PREFIX/pkgs
EOF

# setup area
echo setting up conda environment
module load $(pwd)/modulefile
conda config --remove channels intel

echo CONDA BINARY: $(which conda)
echo CONDA VERSION: $(conda --version)



echo install tensorflow 

# install tensorflow depenedencies
conda install -y tensorflow 

# install pytorch
echo install pytorch
conda install -y pytorch torchvision

echo install other tools
conda install -y scikit-learn scikit-image pandas matplotlib h5py  

echo install horovod
conda install -y -c alcf-theta horovod


