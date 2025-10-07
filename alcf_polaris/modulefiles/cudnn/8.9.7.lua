help([[
Loads cuDNN 8.9.7.29 library for CUDA 12.x
]])

whatis("Name: cuDNN")
whatis("Version: 8.9.7.28")
whatis("Target: x86_64")
whatis("Category: backend, machine learning, NVIDIA")
whatis("Keywords: NVIDIA, ml, machine learning, primitives, nlp, cv, ai, deep learning")
whatis("Description: NVIDIA cuDNN is a GPU-accelerated library of primitives for deep neural networks.")
whatis("URL: https://developer.nvidia.com/cudnn-downloads?target_os=Linux&target_arch=x86_64&Distribution=Agnostic&cuda_version=12")
-- whatis("URL: https://developer.nvidia.com/rdp/cudnn-archive")

local cudnn_dir = "/soft/libraries/cudnn/cudnn-cuda12-linux-x64-v8.9.7.29/"
setenv("CUDNN_HOME",cudnn_dir)
prepend_path("LD_LIBRARY_PATH",pathJoin(cudnn_dir,"lib/"))

-- KGF: user might want to use cuDNN library with Cray PE system nvhpc 
-- prereq(atleast("cudatoolkit-standalone","12.0"))
-- depends_on(atleast("cudatoolkit-standalone","12.0"))  -- cudatoolkit-standalone/12.2.2 is default set via .modulerc.lua
-- depends_on(cudatoolkit-standalone/12.4.1)
