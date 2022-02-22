# courtesy of Victor Gensini- 2022-02-20

# create env from the base anaconda dist
conda create --name wrfpost python=3.9

# activate env
conda activate wrfpost

# install needed libraries for netcdf4
conda install -c conda-forge cython numpy cftime setuptools

# (ensure LD_LIBRARY_PATH is set; I manually did this in my .bashrc file)

# install mpi4py; can test interactively using aprun -n 8 -N 4 python -m mpi4py.bench helloworld
env MPICC=cc pip install mpi4py --no-cache-dir

# clone the netcdf4-python repo
git clone https://github.com/Unidata/netcdf4-python.git

# head into the repo dir
cd netcdf4-python/

# build netcdf4-python
USE_NCCONFIG=0 USE_SETUPCFG=0 NETCDF4_LIBDIR=/opt/cray/pe/netcdf-hdf5parallel/default/INTEL/19.1/lib NETCDF4_INCDIR=/opt/cray/pe/netcdf-hdf5parallel/default/INTEL/19.1/include HDF5_LIBDIR=/opt/cray/pe/hdf5-parallel/default/INTEL/19.1/lib HDF5_INCDIR=/opt/cray/pe/hdf5-parallel/default/INTEL/19.1/include MPICC=cc CC=cc CXX=cc python setup.py build

# install netcdf4-python
USE_NCCONFIG=0 USE_SETUPCFG=0 NETCDF4_LIBDIR=/opt/cray/pe/netcdf-hdf5parallel/default/INTEL/19.1/lib NETCDF4_INCDIR=/opt/cray/pe/netcdf-hdf5parallel/default/INTEL/19.1/include HDF5_LIBDIR=/opt/cray/pe/hdf5-parallel/default/INTEL/19.1/lib HDF5_INCDIR=/opt/cray/pe/hdf5-parallel/default/INTEL/19.1/include MPICC=cc CC=cc CXX=cc python setup.py install

# (can test using https://github.com/Unidata/netcdf4-python/blob/master/examples/mpi_example.py)
#qsub -q debug-flat-quad -n 2 -t 30 -I -A climate_severe
#conda activate wrfpost
#aprun -n 4 -N 4 python mpi_example.py
# (will produce parallel_test.nc if successful)
