help([[
The Anaconda python environment.
Includes build of TensorFlow, PyTorch, DeepHyper, Horovd from tagged versions or develop/master branch of the git repos
DeepHyper version tag: 4717fbe4
TensorFlow version tag: post-2.7.0 (58b34c6c)
Horovod version tag: 0.23.0
PyTorch version tag: 1.10.0a0+git36449ea
- PyTorch DDP with NCCL should be fully functional
- PyTorch with Magma, LAPACK support installed
Mamba package manager installed

You can modify this environment as follows:

  - Extend this environment locally

      $ pip install --user [package]

  - Create a new one of your own

      $ conda create -n [environment_name] [package]

https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html
]])

whatis("Name: conda")
whatis("Version: 4.11.0")
whatis("Category: python conda")
whatis("Keywords: python conda")
whatis("Description: Base Anaconda python environment")
whatis("URL: https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html")

depends_on("openmpi/openmpi-4.1.4_ucx-1.12.1_gcc-9.4.0")
-- KGF: probably don't need this; using copy in /lus/theta-fs0/software/thetagpu/cuda/nccl_2.11.4-1+cuda11.4_x86_64
-- depends_on("nccl/nccl-v2.11.4-1_CUDA11.4")


local conda_dir = "/lus/theta-fs0/software/thetagpu/conda/2021-11-30/mconda3"
local funcs = "conda __conda_activate __conda_hashr __conda_reactivate __add_sys_prefix_to_path"
local home = os.getenv("HOME")

-- Specify where system and user environments should be created
-- setenv("CONDA_ENVS_PATH", pathJoin(conda_dir,"envs"))
-- Directories are separated with a comma
-- setenv("CONDA_PKGS_DIRS", pathJoin(conda_dir,"pkgs"))
-- set environment name for prompt tag
setenv("ENV_NAME",myModuleFullName())
setenv("PYTHONUSERBASE",pathJoin(home,".local/",myModuleFullName()))
setenv("PYTHONSTARTUP",pathJoin(conda_dir,"etc/pythonstart"))

-- add cuda libraries
prepend_path("LD_LIBRARY_PATH","/usr/local/cuda-11.4/lib64")
prepend_path("LD_LIBRARY_PATH","/lus/theta-fs0/software/thetagpu/cuda/cudnn-11.4-linux-x64-v8.2.4.15/lib64")
prepend_path("PATH","/lus/theta-fs0/software/thetagpu/cuda/nccl_2.11.4-1+cuda11.4_x86_64/include")
prepend_path("LD_LIBRARY_PATH","/lus/theta-fs0/software/thetagpu/cuda/nccl_2.11.4-1+cuda11.4_x86_64/lib")
prepend_path("LD_LIBRARY_PATH","/lus/theta-fs0/software/thetagpu/cuda/TensorRT-8.2.1.8.Linux.x86_64-gnu.cuda-11.4.cudnn8.2/lib")

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
