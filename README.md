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
- [ ] Add Mxnet
- [ ] Move future conda environments from Python 3.8 to 3.9 (requirement for HPE Dragon e.g.)
- [ ] `conda-forge` just has `numpy`, non-metapackage? No `numpy-base`, unlike `defaults`? https://stackoverflow.com/questions/50699252/anaconda-environment-installing-packages-numpy-base
- [ ] Why does ThetaGPU seem to demand an OpenMPI/UCX module built against CUDA 11.8 and not 11.4 when TF/Torch/etc. built with 11.8, yet Cray MPICH on Polaris doesnt seem to care about the minor version of CUDA loaded at runtime and used to build the deep learning libraries?
- [ ] Double check that `rpath` solution to DeepSpeed dynamic linking to `libaio` is working
- [ ] Why does `pip install sdv>=0.17.1` reinstall numpy everywhere, and also breaks torch, installs other junk on Polaris? numpy 1.24.1 ---> 1.22.4 on ThetaGPU, even though the existing version seems to match???
```
Collecting numpy<2,>=1.20.0
  Downloading numpy-1.22.4-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (16.9 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 16.9/16.9 MB 228.4 MB/s eta 0:00:00
       Attempting uninstall: numpy
    Found existing installation: numpy 1.24.1
    Uninstalling numpy-1.24.1:
      Successfully uninstalled numpy-1.24.1
```      
- [ ] Consider fixes for problems arising from mixing `conda-forge` and `defaults` packages. **Edit:** now trying `conda install -c defaults -c conda-forge ...` on the one line

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

- When did I start adding things from `conda-forge`? 
  - **Answer**: `mamba` might be the only package actually needed that is not on main `anaconda/defaults` channel. 
- When did I retroactively add `python-libaio` to existing environments? https://github.com/vpelletier/python-libaio
  - **Answer**: `python-libaio` is also example of a package not on `defaults`, that is on `conda-forge`. But for DeepSpeed built from source, we might only need `libaio`, which is on defaults. Sam requested `python-libaio` on 2022-11-09, but I dont think it was ever installed via these scripts or retroactively in existing conda environments (he was experimenting in a clone).
