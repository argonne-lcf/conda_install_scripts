#!/bin/bash

CONDAVER=3
VERSION=4.4.10
BASE_DIR=$PWD/miniconda$CONDAVER

echo Installing into $BASE_DIR
read -p "Are you sure? " -n 1 -r
echo  
if [[ $REPLY =~ ^[Yy]$ ]]
then
    echo Installing
else
   exit -1
fi

#BASE_DIR=/tmp/conda/miniconda$CONDAVER
PREFIX_PATH=$BASE_DIR/$VERSION
DOWNLOAD_PATH=$BASE_DIR/DOWNLOADS

mkdir -p $PREFIX_PATH
mkdir -p $DOWNLOAD_PATH

echo "Downloading miniconda installer from  https://repo.continuum.io/miniconda/Miniconda$CONDAVER-$VERSION-Linux-x86_64.sh"
wget https://repo.continuum.io/miniconda/Miniconda$CONDAVER-$VERSION-Linux-x86_64.sh -P $DOWNLOAD_PATH

chmod +x $DOWNLOAD_PATH/Miniconda$CONDAVER-$VERSION-Linux-x86_64.sh

echo "Installing Miniconda: ./$DOWNLOAD_PATH/Miniconda$CONDAVER-$VERSION-Linux-x86_64.sh -b -p $PREFIX_PATH -u"
$DOWNLOAD_PATH/Miniconda$CONDAVER-$VERSION-Linux-x86_64.sh -b -p $PREFIX_PATH -u

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

# install tensorflow depenedencies
conda install -y gflags glog numpy mkl-dnn scipy pandas h5py virtualenv protobuf grpcio funcsigs pbr mock html5lib bleach werkzeug markdown tensorboard=1.6.0 gast absl-py backports.weakref termcolor astor scikit-learn mpi4py

echo copy MPICH libs to local area
cp /opt/cray/pe/mpt/default/gni/mpich-intel-abi/16.0/lib/libmpi*  ./lib/

# install tensorflow
echo installing tensorflow for python version $PYTHON_VER
if [ "$PYTHON_VER" = "3.6" ]; then
   pip install https://anaconda.org/intel/tensorflow/1.6.0/download/tensorflow-1.6.0-cp36-cp36m-linux_x86_64.whl
elif [ "$PYTHON_VER" = "2.7" ]; then
   pip install https://anaconda.org/intel/tensorflow/1.6.0/download/tensorflow-1.6.0-cp27-cp27mu-linux_x86_64.whl
else
   echo no tensorflow for this python version
fi

# install keras and horovod
pip install keras horovod

