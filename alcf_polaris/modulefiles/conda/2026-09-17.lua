help([[
The Anaconda python environment.
Includes from-source builds of TensorFlow and PyTorch, plus a full inference
stack (mamba-ssm, TransformerEngine, vLLM, FlashInfer, flash-attn, verl) and
JAX from binary wheels.

TensorFlow version tag: 2.21.0 (built from source, hermetic CUDA/XLA)
PyTorch version tag:    2.14.0 (built from source, CUDA 13.0)
Python version:         3.13

You can modify this environment as follows:

  - Extend this environment locally

      $ pip install --user [package]

  - Create a new one of your own

      $ conda create -n [environment_name] [package]

https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html
]])

whatis("Name: conda")
-- note: Miniforge installer often lags behind conda binary version, which is
-- updated in the install script. Verify with `conda --version` after loading.
whatis("Version: 26.7.2-0 miniforge; 26.7.2 conda, conda-build versions")
whatis("Category: python conda")
whatis("Keywords: python conda")
whatis("Description: Base miniforge Python environment")
whatis("URL: https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html")

depends_on("PrgEnv-gnu")
depends_on("craype-x86-milan")
depends_on("cray-hdf5-parallel/1.14.3.9")
-- PE 26.03: gcc-native/14 is 14.3.0 (== /usr/bin/gcc-14); gcc-native/14.2 is gone.
depends_on("gcc-native/14")
-- No cudnn/ modulefile for the cuda13 9.26 build yet; paths are set directly below.
-- note, unloading this does not remove /usr/bin/g++-14; just means /usr/bin/g++
-- (7.5.0) is the first in PATH, not /opt/cray/pe/gcc-native/14/bin/g++.

-- helps when vLLM JIT compiles things:
setenv("CC","/usr/bin/gcc-14")
setenv("CXX","/usr/bin/g++-14")

setenv("TORCH_CUDA_ARCH_LIST","8.0")
setenv("FLASHINFER_CUDA_ARCH_LIST","8.0")

local base_path = "/soft/applications/conda/2026-09-17/"
setenv("BASE_PATH",base_path)
local conda_dir = pathJoin(base_path,"mconda3")
local funcs = "conda __conda_activate __conda_hashr __conda_reactivate"
local home = os.getenv("HOME")

-- Specify where system and user environments should be created
-- setenv("CONDA_ENVS_PATH", pathJoin(conda_dir,"envs"))
-- Directories are separated with a comma
-- setenv("CONDA_PKGS_DIRS", pathJoin(conda_dir,"pkgs"))

-- set environment name for prompt tag
setenv("ENV_NAME",myModuleFullName())
local pyuserbase = pathJoin(home,".local/","polaris/",myModuleFullName())

setenv("PYTHONUSERBASE", pyuserbase)
unsetenv("PYTHONSTARTUP") -- ,pathJoin(conda_dir,"etc/pythonstart"))

-- KGF: could add this, but "conda activate" will put "/soft/datascience/conda/date/mconda3/bin" ahead of it
-- Alternative is to "export PATH=$PYTHONUSERBASE/bin:$PATH" in mconda3/etc/conda/activate.d/env_vars.sh (and undo in deactivate.d/)
-- prepend_path("PATH",pathJoin(pyuserbase, "bin/"))

-- add cuda libraries (all CUDA 13 builds; see build_monolithic_conda_module.sh header)
local cudnn_home = "/soft/libraries/cudnn/cudnn-cuda13-linux-x64-v9.26.0.51/"
setenv("CUDNN_BASE",cudnn_home)
prepend_path("LD_LIBRARY_PATH",pathJoin(cudnn_home,"lib/"))
prepend_path("CPATH",pathJoin(cudnn_home,"include/"))
local nccl_home = "/soft/libraries/nccl/nccl_2.30.7-1+cuda13.3_x86_64/"
setenv("NCCL_BASE",nccl_home)
setenv("NCCL_HOME",nccl_home)
prepend_path("LD_LIBRARY_PATH",pathJoin(nccl_home,"lib/"))
prepend_path("CPATH",pathJoin(nccl_home,"include/"))
prepend_path("LD_LIBRARY_PATH","/soft/libraries/trt/TensorRT-10.16.1.11.Linux.x86_64-gnu.cuda-13.2/lib")
prepend_path("LD_LIBRARY_PATH","/soft/libraries/cusparselt/libcusparse_lt-linux-x86_64-0.9.1.1_cuda13-archive/lib")

local cuda_home = "/soft/compilers/cudatoolkit/cuda-13.0.3/"
setenv("CUDA_HOME",cuda_home)
setenv("CUDA_PATH",cuda_home)  -- KeOps
setenv("CUDA_TOOLKIT_BASE",cuda_home)
prepend_path("PATH",pathJoin(cuda_home,"bin/"))
prepend_path("LD_LIBRARY_PATH",pathJoin(cuda_home,"lib64/"))
-- CUPTI:
prepend_path("LD_LIBRARY_PATH",pathJoin(cuda_home,"extras/CUPTI/lib64/"))
-- TransformerEngine <= 2.7 needed this to import; harmless on 2.19, kept for parity
-- with the 2025-09 modules.
setenv("NVTE_CUDA_INCLUDE_DIR",pathJoin(cuda_home,"include/"))

-- DeepSpeed libaio
setenv("CFLAGS","-I" .. pathJoin(conda_dir,"include/"))
setenv("LDFLAGS","-L" .. pathJoin(conda_dir,"lib/") .. " -Wl,--enable-new-dtags,-rpath," .. pathJoin(conda_dir,"lib/"))

setenv("https_proxy","http://proxy.alcf.anl.gov:3128")
setenv("http_proxy","http://proxy.alcf.anl.gov:3128")

-- Enable CUDA-aware MPICH, by default
setenv("MPICH_GPU_SUPPORT_ENABLED",1)

-- (mpi4)Jax/TensorFlow/XLA flags:
setenv("MPI4JAX_USE_CUDA_MPI",1)
-- first flag is Jax workaround, second flag is TF workaround when CUDA Toolkit is moved after installation
-- (XLA hardcodes location to CUDA https://github.com/tensorflow/tensorflow/issues/23783)
setenv("XLA_FLAGS","--xla_gpu_force_compilation_parallelism=1 --xla_gpu_cuda_data_dir=" .. cuda_home)
-- Corey: pretty sure the following flag isnt working for Jax
setenv("XLA_PYTHON_CLIENT_PREALLOCATE","false")

-- Initialize conda
execute{cmd="source " .. conda_dir .. "/etc/profile.d/conda.sh;", modeA={"load"}}
execute{cmd="[[ -z ${ZSH_EVAL_CONTEXT+x} ]] && export -f " .. funcs, modeA={"load"}}
-- Unload environments and clear conda from environment
execute{cmd="for i in $(seq ${CONDA_SHLVL:=0}); do conda deactivate; done; pre=" .. conda_dir .. "; \
	export LD_LIBRARY_PATH=$(echo ${LD_LIBRARY_PATH} | tr ':' '\\n' | grep . | grep -v $pre | tr '\\n' ':' | sed 's/:$//'); \
	export PATH=$(echo ${PATH} | tr ':' '\\n' | grep . | grep -v $pre | tr '\\n' ':' | sed 's/:$//'); \
	unset -f " .. funcs .. "; \
	unset $(env | grep -o \"[^=]*CONDA[^=]*\");", modeA={"unload"}}

-- Prevent from being loaded with another system python or conda environment
family("python")
unload("xalt")

-- No pe-26.03-shim dir here: this env was built on PE 26.03 / CUDA 13, so every
-- MPI-linked extension (mpi4py, h5py, torch, mpi4jax) already links libmpi_gtl_cuda
-- and libmpi_gnu.so.12, and the GTL's libcudart.so.13 matches the toolkit.
