# conda_install_scripts
## Usage
### Polaris

```
qsub -A datascience -q preemptable -l select=1:ncpus=64:ngpus=4,filesystems=swift,walltime=02:30:00 -joe -- install_datascience_conda_fromsource.sh /soft/datascience/conda/2023-01-10
```

### ThetaGPU
```
qsub -t 170 -n 1 -q full-node -A datascience -M <email> -o build.out -e build.out ./install_datascience_conda.sh /lus/theta-fs0/software/thetagpu/conda/2023-01-11
```

## Dependencies 
**How to download the CUDA Toolkit from NVIDIA and extract just the toolkit, not installing the driver, etc. without `sudo` permissions:**

E.g. on Polaris, download from NVIDIA, selecting Target Platform:
- Linux > x86_64 > SLES > Version 15 > runfile (local) =
- https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=SLES&target_version=15&target_type=runfile_local
```
cd /soft/compilers/cudatoolkit
wget https://developer.download.nvidia.com/compute/cuda/11.7.1/local_installers/cuda_11.7.1_515.65.01_linux.run
sh cuda_11.7.1_515.65.01_linux.run --silent --toolkit --toolkitpath=$PWD/cuda-11.7.1
```
## To-do
- [ ] Create bash scripts for testing environments based on https://anl.app.box.com/notes/1001252052445
- [ ] ThetaGPU script is not installing parallel h5py like in Polaris script
- [ ] Move future conda environments from Python 3.8 to 3.9 (requirement for HPE Dragon e.g.)
- [ ] `conda-forge` just has `numpy`, non-metapackage? No `numpy-base`, unlike `defaults`? https://stackoverflow.com/questions/50699252/anaconda-environment-installing-packages-numpy-base
- [ ] Consider fixes for problems arising from mixing `conda-forge` and `defaults` packages

https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-channels.html

https://conda-forge.org/docs/user/tipsandtricks.html
```
conda config --set channel_priority strict
```

https://stackoverflow.com/questions/65483245/how-to-avoid-using-conda-forge-packages-unless-necessary
```
conda install -c defaults -c conda-forge somepackage
```
which puts defaults with top priority. Or:
```
conda install conda-forge::somepackage
```
and this will not change the channel priority.



Also, interestingly:
> To solve these issues, conda-forge has created special dummy builds of the mpich and openmpi libraries that are simply shell packages with no contents. These packages allow the conda solver to produce correct environments while avoiding installing MPI binaries from conda-forge. You can install the dummy package with the following command
```
$ conda install "mpich=x.y.z=external_*"
$ conda install "openmpi=x.y.z=external_*"
```

- When did I start adding things from `conda-forge`? `python-libaio` is an example of a package not on `defaults`
- When did I retroactively add `python-libaio` to existing environments? https://github.com/vpelletier/python-libaio
