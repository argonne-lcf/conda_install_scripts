help([[
The Anaconda python environment.
Includes from-source builds of TensorFlow and PyTorch, plus a full inference
stack (mamba-ssm, TransformerEngine, vLLM, SGLang, FlashInfer, flash-attn,
verl) and JAX from binary wheels.

TensorFlow version tag: 2.21.0 (built from source, hermetic CUDA/XLA)
PyTorch version tag:    2.12.0 (built from source)
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
whatis("Version: 26.5.2")
whatis("Category: python conda")
whatis("Keywords: python conda")
whatis("Description: Base Anaconda python environment")
whatis("URL: https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html")

depends_on("compilers/clang/release-22.1.0")
depends_on("compilers/openmpi/5.0.10")
depends_on("hdf5/2.1.1-openmpi-5.0.10")

local base_path = "/soft/applications/conda/2026-06-08/"
setenv("BASE_PATH", base_path)
local conda_dir = pathJoin(base_path, "mconda3")
local funcs = "conda __conda_activate __conda_hashr __conda_reactivate"
local home = os.getenv("HOME")

-- Specify where system and user environments should be created
-- setenv("CONDA_ENVS_PATH", pathJoin(conda_dir,"envs"))
-- Directories are separated with a comma
-- setenv("CONDA_PKGS_DIRS", pathJoin(conda_dir,"pkgs"))
-- set environment name for prompt tag
setenv("ENV_NAME",myModuleFullName())

-- Sophia is DGX A100 (sm_80 only). Pin runtime JIT compilers to this arch so
-- flash-attn / vLLM / Triton / cpp_extension don't fan out to all visible
-- archs (the same OOM hazard we hit at build time).
setenv("TORCH_CUDA_ARCH_LIST","8.0")
setenv("FLASHINFER_CUDA_ARCH_LIST","8.0")

-- depends_on openmpi sets CC=mpicc / CXX=mpicxx, which silently links
-- libmpi.so (and its NEEDED chain) into JIT-built torch extensions
-- (vLLM custom kernels, Triton autotune, torch.utils.cpp_extension).
-- Pin plain system gcc/g++ so runtime JIT compiles stay MPI-free.
setenv("CC","/usr/bin/gcc")
setenv("CXX","/usr/bin/g++")

local pyuserbase = pathJoin(home,".local/","sophia/",myModuleFullName())

setenv("PYTHONUSERBASE", pyuserbase)
unsetenv("PYTHONSTARTUP") -- ,pathJoin(conda_dir,"etc/pythonstart"))

-- KGF: could add this, but "conda activate" will put "/soft/datascience/conda/2022-07-19/mconda3/bin" ahead of it
-- Alternative is to "export PATH=$PYTHONUSERBASE/bin:$PATH" in mconda3/etc/conda/activate.d/env_vars.sh (and undo in deactivate.d/)
-- prepend_path("PATH",pathJoin(pyuserbase, "bin/"))

-- add cuda libraries (versions tracked in build_monolithic_conda_module.sh)
prepend_path("LD_LIBRARY_PATH","/soft/libraries/cudnn/cudnn-cuda13-linux-x64-v9.22.0.52/lib")
prepend_path("PATH","/soft/libraries/nccl/nccl_2.30.4-1+cuda13.2_x86_64/include")
prepend_path("LD_LIBRARY_PATH","/soft/libraries/nccl/nccl_2.30.4-1+cuda13.2_x86_64/lib")
prepend_path("LD_LIBRARY_PATH","/soft/libraries/trt/TensorRT-10.16.1.11/lib")
-- cuSPARSELt (new in the 2026 build; consumed by PyTorch via USE_CUSPARSELT=1)
prepend_path("LD_LIBRARY_PATH","/soft/libraries/cusparselt/libcusparse_lt-linux-x86_64-0.9.1.1_cuda13-archive/lib")

local cuda_home = "/soft/compilers/cudatoolkit/cuda-13.2.1/"
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
