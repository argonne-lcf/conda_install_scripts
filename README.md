# conda_install_scripts

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

- [ ] ThetaGPU script is not installing parallel h5py like in Polaris script
