help([[
The Anaconda python environment.
Includes build of TensorFlow, PyTorch, DeepHyper, Horovd from tagged versions or develop/master branch of the git repos
DeepHyper version: 901eb2d478 (Dec 2022) [analytics,hvd,nas,popt,autodeuq,svd]
TensorFlow version tag: 2.11.0
Horovod version tag: 0.26.1
PyTorch version tag: 1.13.1

You can modify this environment as follows:

  - Extend this environment locally

      $ pip install --user [package]

  - Create a new one of your own

      $ conda create -n [environment_name] [package]

https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html
]])

whatis("Name: conda")
whatis("Version: 22.11.1")
whatis("Category: python conda")
whatis("Keywords: python conda")
whatis("Description: Base Anaconda python environment")
whatis("URL: https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html")

depends_on("openmpi/openmpi-4.1.4_ucx-1.14.0_gcc-9.4.0_cuda-11.8")


local conda_dir = "/lus/theta-fs0/software/thetagpu/conda/2023-01-11/mconda3"
local funcs = "conda __conda_activate __conda_hashr __conda_reactivate"
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
-- local cuda_home = "/usr/local/cuda-11.4"
local cuda_home = "/lus/theta-fs0/software/thetagpu/cuda/cuda-11.8.0"
setenv("CUDA_HOME",cuda_home)
prepend_path("PATH",pathJoin(cuda_home,"bin/"))
prepend_path("LD_LIBRARY_PATH",pathJoin(cuda_home,"lib64/"))
-- CUPTI:
prepend_path("LD_LIBRARY_PATH",pathJoin(cuda_home,"extras/CUPTI/lib64/"))

prepend_path("LD_LIBRARY_PATH","/lus/theta-fs0/software/thetagpu/cuda/cudnn-linux-x86_64-8.6.0.163_cuda11-archive/lib")
prepend_path("PATH","/lus/theta-fs0/software/thetagpu/cuda/nccl_2.16.2-1+cuda11.8_x86_64/include")
prepend_path("LD_LIBRARY_PATH","/lus/theta-fs0/software/thetagpu/cuda/nccl_2.16.2-1+cuda11.8_x86_64/lib")
prepend_path("LD_LIBRARY_PATH","/lus/theta-fs0/software/thetagpu/cuda/TensorRT-8.5.2.2/lib")

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
