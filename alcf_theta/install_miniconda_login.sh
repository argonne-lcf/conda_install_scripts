#!/bin/bash

CONDAVER=2
VERSION=4.5.4
BASE_DIR=$PWD/miniconda$CONDAVER
PREFIX_PATH=$BASE_DIR/$VERSION-login
DOWNLOAD_PATH=$BASE_DIR/DOWNLOADS

echo Installing into $PREFIX_PATH
read -p "Are you sure? " -n 1 -r
echo  
if [[ $REPLY =~ ^[Yy]$ ]]
then
    echo Installing
else
   exit -1
fi

umask 0022

mkdir -p $PREFIX_PATH
mkdir -p $DOWNLOAD_PATH

MINICONDA_INSTALL_FILE=Miniconda$CONDAVER-$VERSION-Linux-x86_64.sh

if [ ! -f $DOWNLOAD_PATH/$MINICONDA_INSTALL_FILE ]; then
   echo Downloading miniconda installer
   wget https://repo.continuum.io/miniconda/$MINICONDA_INSTALL_FILE -P $DOWNLOAD_PATH
   
   chmod +x $DOWNLOAD_PATH/Miniconda$CONDAVER-$VERSION-Linux-x86_64.sh
fi

echo Installing Miniconda
$DOWNLOAD_PATH/Miniconda$CONDAVER-$VERSION-Linux-x86_64.sh -b -p $PREFIX_PATH -u

echo moving into $PREFIX_PATH
cd $PREFIX_PATH

PYTHON_VER=$(ls -d lib/python?.? | tail -c4)
echo PYTHON_VER=$PYTHON_VER

# create a setup file
cat > setup.sh << EOF
DIR=\$( cd "\$( dirname "\${BASH_SOURCE[0]}" )" && pwd )
export LD_LIBRARY_PATH=\$DIR/lib:\$LD_LIBRARY_PATH
export PATH=\$DIR/bin:\$PATH
export PYTHONPATH=\$DIR/lib/python3.6/site-packages:\$PYTHONPATH
export PYTHONSTARTUP=\$PWD/etc/pythonstart
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
## miniconda3 modulefile
##
proc ModulesHelp { } {
   global CONDA_LEVEL PYTHON_LEVEL MINICONDA_LEVEL
   puts stderr "This module will add Miniconda \$MINICONDA_LEVEL to your environment with conda version \$CONDA_LEVEL and python version \$PYTHON_LEVEL"
}

set _module_nme   [module-info name]
set is_module_rm  [module-info mode remove]
set sys           [uname sysname]
set os            [uname release]

set PYTHON_LEVEL                 $PYTHON_VER
set CONDA_LEVEL                  $VERSION
set MINICONDA_LEVEL              $CONDAVER
set MINICONDA_INSTALL_PATH       $PREFIX_PATH
setenv PYTHONSTARTUP             \$MINICONDA_INSTALL_PATH/etc/pythonstart

prepend-path   PATH              \$MINICONDA_INSTALL_PATH/bin
prepend-path   LD_LIBRARY_PATH   \$MINICONDA_INSTALL_PATH/lib
prepend-path   PYTHONPATH        \$MINICONDA_INSTALL_PATH/lib/python$PYTHON_VER/site-packages

module-whatis  "miniconda installation"
EOF

# setup area
echo setting up conda environment
module load $(pwd)/modulefile

echo CONDA BINARY: $(which conda)
echo CONDA VERSION: $(conda --version)



echo install tensorflow dependencies and other things

conda install -y pytorch gflags glog numpy scipy pandas h5py virtualenv protobuf grpcio funcsigs pbr mock html5lib bleach werkzeug markdown gast absl-py backports.weakref termcolor astor scikit-learn mpi4py 
conda install -y -c conda-forge mpich
# install keras and horovod
pip install --no-deps tensorflow tensorboard
pip install keras horovod

