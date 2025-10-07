help([[
Loads cuDNN 9.12.0.46 library for CUDA 12.x
]])

whatis("Name: cuDNN")
whatis("Version: 9.12.0.46")
whatis("Target: x86_64")
whatis("Category: backend, machine learning, NVIDIA")
whatis("Keywords: NVIDIA, ml, machine learning, primitives, nlp, cv, ai, deep learning")
whatis("Description: NVIDIA cuDNN is a GPU-accelerated library of primitives for deep neural networks.")
whatis("URL: https://developer.nvidia.com/cudnn-downloads?target_os=Linux&target_arch=x86_64&Distribution=Agnostic&cuda_version=12")
-- whatis("URL: https://developer.nvidia.com/rdp/cudnn-archive")

local cudnn_dir = "/soft/libraries/cudnn/cudnn-cuda12-linux-x64-v9.12.0.46/"
setenv("CUDNN_HOME",cudnn_dir)
prepend_path("LD_LIBRARY_PATH",pathJoin(cudnn_dir,"lib/"))
