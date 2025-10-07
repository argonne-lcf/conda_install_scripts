help([[
The Anaconda python environment.
Limited environment built for 2024 ALCF Hands-on HPC Workshop
AI Inference Optimization tutorial
PyTorch version tag: 2.5.0
vLLM
Horovod version tag: 3a31d933a

You can modify this environment as follows:

  - Extend this environment locally

      $ pip install --user [package]

  - Create a new one of your own

      $ conda create -n [environment_name] [package]

https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html
]])

whatis("Name: conda")
-- note, miniconda installer often lags behind conda binary version, which is updated in the install script
whatis("Version: 24.9.2")
whatis("Category: python conda")
whatis("Keywords: python conda")
whatis("Description: Base Anaconda python environment")
whatis("URL: https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html")

depends_on("PrgEnv-gnu")
depends_on("craype-x86-milan")
depends_on("cray-hdf5-parallel/1.12.2.9")
depends_on("cudnn/9.4.0")

local conda_dir = "/soft/applications/conda/2024-10-30-workshop/mconda3"
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

-- KGF: could add this, but "conda activate" will put "/soft/datascience/conda/2022-07-19/mconda3/bin" ahead of it
-- Alternative is to "export PATH=$PYTHONUSERBASE/bin:$PATH" in mconda3/etc/conda/activate.d/env_vars.sh (and undo in deactivate.d/)
-- prepend_path("PATH",pathJoin(pyuserbase, "bin/"))

-- add cuda libraries
prepend_path("PATH","/soft/libraries/nccl/nccl_2.23.4-1+cuda12.6_x86_64/include")
prepend_path("LD_LIBRARY_PATH","/soft/libraries/nccl/nccl_2.23.4-1+cuda12.6_x86_64/lib")
prepend_path("LD_LIBRARY_PATH","/soft/libraries/trt/TensorRT-10.4.0.26.Linux.x86_64-gnu.cuda-12.5/lib")

local cuda_home = "/soft/compilers/cudatoolkit/cuda-12.6.1/"
setenv("CUDA_HOME",cuda_home)
setenv("CUDA_PATH",cuda_home)  -- KeOps
setenv("CUDA_TOOLKIT_BASE",cuda_home)
prepend_path("PATH",pathJoin(cuda_home,"bin/"))
prepend_path("LD_LIBRARY_PATH",pathJoin(cuda_home,"lib64/"))
-- CUPTI:
prepend_path("LD_LIBRARY_PATH",pathJoin(cuda_home,"extras/CUPTI/lib64/"))

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

-- Huihuo: optimized NCCL settings for PyTorch performance, October 2024:
-- setenv("NCCL_NET_GDR_LEVEL","PHB")
-- setenv("NCCL_CROSS_NIC",1)
-- setenv("NCCL_COLLNET_ENABLE",1)
-- setenv("NCCL_NET","AWS Libfabric")
-- setenv("FI_CXI_DISABLE_HOST_REGISTER",1)
-- setenv("FI_MR_CACHE_MONITOR","userfaultfd")
-- setenv("FI_CXI_DEFAULT_CQ_SIZE",131072)

-- prepend_path("LD_LIBRARY_PATH","/soft/libraries/aws-ofi-nccl/v1.9.1-aws/lib")
-- prepend_path("LD_LIBRARY_PATH","/soft/libraries/hwloc/lib/")

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
