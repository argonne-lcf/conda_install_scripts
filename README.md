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
