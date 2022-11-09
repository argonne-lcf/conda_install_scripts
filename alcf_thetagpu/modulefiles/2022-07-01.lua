help([[
The Anaconda python environment.
Includes build of TensorFlow, PyTorch, DeepHyper, Horovd from tagged versions or develop/master branch of the git repos
DeepHyper version tag: 0.4.2
TensorFlow version tag: 2.9.1
Horovod version tag: 0.25.0
PyTorch version tag: 1.12.0

You can modify this environment as follows:

  - Extend this environment locally

      $ pip install --user [package]

  - Create a new one of your own

      $ conda create -n [environment_name] [package]

https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html
]])

whatis("Name: conda")
whatis("Version: 4.13.0")
whatis("Category: python conda")
whatis("Keywords: python conda")
whatis("Description: Base Anaconda python environment")
whatis("URL: https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html")

depends_on("openmpi/openmpi-4.1.4_ucx-1.12.1_gcc-9.4.0")


local conda_dir = "/lus/theta-fs0/software/thetagpu/conda/2022-07-01/mconda3"
local funcs = "conda __conda_activate __conda_hashr __conda_reactivate __add_sys_prefix_to_path"
local home = os.getenv("HOME")

-- Specify where system and user environments should be created
-- setenv("CONDA_ENVS_PATH", pathJoin(conda_dir,"envs"))
-- Directories are separated with a comma
-- setenv("CONDA_PKGS_DIRS", pathJoin(conda_dir,"pkgs"))
-- set environment name for prompt tag
setenv("ENV_NAME",myModuleFullName())
setenv("PYTHONUSERBASE",pathJoin(home,".local/","thetagpu",myModuleFullName()))
unsetenv("PYTHONSTARTUP") -- ,pathJoin(conda_dir,"etc/pythonstart"))

-- add cuda libraries
local cuda_home = "/usr/local/cuda-11.4"
setenv("CUDA_HOME",cuda_home)
prepend_path("PATH",pathJoin(cuda_home,"bin/"))
prepend_path("LD_LIBRARY_PATH",pathJoin(cuda_home,"lib64/"))
-- CUPTI:
prepend_path("LD_LIBRARY_PATH",pathJoin(cuda_home,"extras/CUPTI/lib64/"))

prepend_path("LD_LIBRARY_PATH","/lus/theta-fs0/software/thetagpu/cuda/cudnn-linux-x86_64-8.4.1.50_cuda11.6-archive/lib64")
prepend_path("PATH","/lus/theta-fs0/software/thetagpu/cuda/nccl_2.12.12-1+cuda11.0_x86_64/include")
prepend_path("LD_LIBRARY_PATH","/lus/theta-fs0/software/thetagpu/cuda/nccl_2.12.12-1+cuda11.6_x86_64/lib")
prepend_path("LD_LIBRARY_PATH","/lus/theta-fs0/software/thetagpu/cuda/TensorRT-8.2.5.1/lib")

setenv("https_proxy","http://proxy.tmi.alcf.anl.gov:3128")
setenv("http_proxy","http://proxy.tmi.alcf.anl.gov:3128")

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
