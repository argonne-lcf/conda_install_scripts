help([[
The Anaconda python environment.
Includes build of TensorFlow, PyTorch, DeepHyper, Horovod from tagged versions or develop/master branch of the git repos
DeepHyper version: 215aa2e (2023-09-21) [analytics,hvd,nas,hps-tl,autodeuq]
TensorFlow version tag: 2.17.0 (wheel, not built from source)
Horovod version tag: 0.28.1, plus C++17 patch
PyTorch version tag: 2.4.0

You can modify this environment as follows:

  - Extend this environment locally

      $ pip install --user [package]

  - Create a new one of your own

      $ conda create -n [environment_name] [package]

https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html
]])

whatis("Name: conda")
-- note, miniconda installer often lags behind conda binary version, which is updated in the install script
whatis("Version: 24.5.0")
whatis("Category: python conda")
whatis("Keywords: python conda")
whatis("Description: Base Anaconda python environment")
whatis("URL: https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html")

depends_on("compilers/clang/release-17.0.6")
depends_on("compilers/openmpi/5.0.3")
depends_on("hdf5/1.14.4.3-openmpi-5.0.3")

local conda_dir = "/soft/applications/conda/2024-08-08/mconda3"
local funcs = "conda __conda_activate __conda_hashr __conda_reactivate"
local home = os.getenv("HOME")

-- Specify where system and user environments should be created
-- setenv("CONDA_ENVS_PATH", pathJoin(conda_dir,"envs"))
-- Directories are separated with a comma
-- setenv("CONDA_PKGS_DIRS", pathJoin(conda_dir,"pkgs"))
-- set environment name for prompt tag
setenv("ENV_NAME",myModuleFullName())
local pyuserbase = pathJoin(home,".local/","sophia/",myModuleFullName())

setenv("PYTHONUSERBASE", pyuserbase)
unsetenv("PYTHONSTARTUP") -- ,pathJoin(conda_dir,"etc/pythonstart"))

-- KGF: could add this, but "conda activate" will put "/soft/datascience/conda/2022-07-19/mconda3/bin" ahead of it
-- Alternative is to "export PATH=$PYTHONUSERBASE/bin:$PATH" in mconda3/etc/conda/activate.d/env_vars.sh (and undo in deactivate.d/)
-- prepend_path("PATH",pathJoin(pyuserbase, "bin/"))

-- add cuda libraries
prepend_path("LD_LIBRARY_PATH","/soft/libraries/cudnn/cudnn-cuda12-linux-x64-v9.3.0.75/lib")
prepend_path("PATH","/soft/libraries/nccl/nccl_2.22.3-1+cuda12.6_x86_64/include")
prepend_path("LD_LIBRARY_PATH","/soft/libraries/nccl/nccl_2.22.3-1+cuda12.6_x86_64/lib")
prepend_path("LD_LIBRARY_PATH","/soft/libraries/trt/TensorRT-8.6.1.6.Linux.x86_64-gnu.cuda-12.0/lib")

local cuda_home = "/soft/compilers/cudatoolkit/cuda-12.6.0/"
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
