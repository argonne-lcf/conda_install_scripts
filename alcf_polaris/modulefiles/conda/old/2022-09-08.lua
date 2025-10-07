help([[
The Anaconda python environment.
Includes build of TensorFlow, PyTorch, DeepHyper, Horovd from tagged versions or develop/master branch of the git repos
DeepHyper version tag: 0.4.2
TensorFlow version tag: 2.10.0
Horovod version tag: 0.25.0
PyTorch version tag: 1.12.1

You can modify this environment as follows:

  - Extend this environment locally

      $ pip install --user [package]

  - Create a new one of your own

      $ conda create -n [environment_name] [package]

https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html
]])

whatis("Name: conda")
-- note, miniconda installer often lags behind conda binary version, which is updated in the install script
whatis("Version: 4.14.0")
whatis("Category: python conda")
whatis("Keywords: python conda")
whatis("Description: Base Anaconda python environment")
whatis("URL: https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html")

depends_on("PrgEnv-gnu")


local conda_dir = "/soft/datascience/conda/2022-09-08/mconda3"
local funcs = "conda __conda_activate __conda_hashr __conda_reactivate __add_sys_prefix_to_path"
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
-- prepend_path("LD_LIBRARY_PATH","/opt/nvidia/hpc_sdk/Linux_x86_64/21.9/cuda/lib64")
prepend_path("LD_LIBRARY_PATH","/soft/libraries/cudnn/cudnn-11.6-linux-x64-v8.4.1.50/lib")
prepend_path("PATH","/soft/libraries/nccl/nccl_2.14.3-1+cuda11.6_x86_64/include")
prepend_path("LD_LIBRARY_PATH","/soft/libraries/nccl/nccl_2.14.3-1+cuda11.6_x86_64/lib")
prepend_path("LD_LIBRARY_PATH","/soft/libraries/trt/TensorRT-8.4.3.1.Linux.x86_64-gnu.cuda-11.6.cudnn8.4/lib")


local cuda_home = "/soft/compilers/cudatoolkit/cuda-11.6.2"
setenv("CUDA_HOME",cuda_home)
prepend_path("PATH",pathJoin(cuda_home,"bin/"))
prepend_path("LD_LIBRARY_PATH",pathJoin(cuda_home,"lib64/"))
-- CUPTI:
prepend_path("LD_LIBRARY_PATH",pathJoin(cuda_home,"extras/CUPTI/lib64/"))

setenv("https_proxy","http://proxy.alcf.anl.gov:3128")
setenv("http_proxy","http://proxy.alcf.anl.gov:3128")

-- Enable CUDA-aware MPICH, by default
setenv("MPICH_GPU_SUPPORT_ENABLED",1)

-- (mpi4)Jax/TensorFlow/XLA flags:
setenv("MPI4JAX_USE_CUDA_MPI",1)
-- setenv("XLA_FLAGS","--xla_gpu_force_compilation_parallelism=1 --xla_gpu_cuda_data_dir=/soft/compilers/cudatoolkit/cuda_11.5.2_495.29.05_linux")
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
