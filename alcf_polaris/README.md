# To do
- [x] Migrate away from `/soft/datascience/conda/2022-07-19-login`
- [x] Monitor fix to `umask` being messed up on compute nodes. Should be 022, not 077
- [x] Confirm PyTorch DDP functionality and performance with Corey
- [x] Re-check NCCL performnace with Denis once the GPUDirect issues are resolved: https://cels-anl.slack.com/archives/C03G5PHHF7V/p1658946362840349
- [x] Get copies of old, pre-AT Tcl Cray PE modulefiles for reference. Not possible, since they are not cached in `/lus/swift/pAT/soft/modulefiles`, but are part of the compute image in `/opt/cray/pe/lmod/modulefiles/core/PrgEnv-nvidia` e.g. 
  - [x] Understand `nvidia_already_loaded` etc. from https://cels-anl.slack.com/archives/GN23NRCHM/p1653506105162759?thread_ts=1653504764.783399&cid=GN23NRCHM (set to 0 in deprecated `PrgEnv-nvidia` and 1 in `PrgEnv-nvhpc` in AT Tcl modulefiles; **removed after AT**)
- [] Get longer-term fix for `PrgEnv-gnu` failure building TF from source:
```
/opt/cray/pe/gcc/11.2.0/bin/redirect: line 5: /opt/cray/pe/gcc/11.2.0/bin/../snos/bin/redirect: No such file or directory
```
**Workaround:** `export GCC_HOST_COMPILER_PATH=/opt/cray/pe/gcc/11.2.0/snos/bin/gcc`

- [ ] Add MXNet, Horovod support?
- [ ] Fix and validate PyTorch+Hvd script with >1 nodes https://github.com/argonne-lcf/dlSoftwareTests/blob/main/pytorch/horovod_mnist.qsub.polaris.sh on Polaris. Works fine on ThetaGPU 2 nodes
- [ ] Suggest and monitor potential changes to new post-AT `cudatoolkit-standalone/11.4.4` etc. Lua modulefiles (Ye Luo wrote them) whereby `#include <cuda_runtime.h>` is not found by the compiler. https://cels-anl.slack.com/archives/GN23NRCHM/p1658958235623699
  - Ti created the Tcl `cudatoolkit` modulefile, during AT (no `CPATH` changes, but `pkgconfig` changes); was automatically loaded with default `PrgEnv-nvidia` (see readme in https://github.com/felker/athenak-scaling/blob/main/results/polaris_scaling.ipynb). See next section for copies of some of the modulefiles. Presumably `pkg-config` automatically modifies the compiler search directories, and/or the system OS `nvidia/22.3` and/or `PrgEnv-nvidia/8.3.3` modulefiles from Cray HPCM (no longer have access to copies of these old modulefiles) somehow modified these directories. I still manually linked
  - Cray HPE provides the Lua `nvhpc` modulefile, post-AT via HPCM (no `pkgconfig` changes; `CPATH` changes do not include base CUDA Toolkit `cuda/include/` subdirectory containing `cuda_runtime.h`:
  ```
  prepend_path("CPATH","/opt/nvidia/hpc_sdk/Linux_x86_64/21.9/math_libs/include")
  prepend_path("CPATH","/opt/nvidia/hpc_sdk/Linux_x86_64/21.9/comm_libs/nccl/include")
  prepend_path("CPATH","/opt/nvidia/hpc_sdk/Linux_x86_64/21.9/comm_libs/nvshmem/include")
  prepend_path("CPATH","/opt/nvidia/hpc_sdk/Linux_x86_64/21.9/compilers/extras/qd/include/qd")
  ```
  - Ye created the Lua `cudatoolkit-standalone`, post-AT (no `CPATH` or `pkgconfig` changes)
  - Perlmutter's `cudatoolkit` modules are distributed by HPE Cray via Shasta CSM (**BOTH** `CPATH` and `pkgconfig` changes):
  
```
setenv("CRAY_CUDATOOLKIT_INCLUDE_OPTS","-I/opt/nvidia/hpc_sdk/Linux_x86_64/21.11/cuda/11.5/include -I/opt/nvidia/hpc_sdk/Linux_x86_64/21.11/cuda/11.5/nvvm/include -I/opt/nvidia/hpc_sdk/Linux_x86_64/21.11/cuda/11.5/extras/Debugger/include -I/opt/nvidia/hpc_sdk/Linux_x86_64/21.11/cuda/11.5/extras/CUPTI/include -I/opt/nvidia/hpc_sdk/Linux_x86_64/21.11/math_libs/11.5/include")

prepend_path("PKG_CONFIG_PATH","/opt/modulefiles/cudatoolkit")
prepend_path("PE_PKGCONFIG_LIBS","cudatoolkit_11.5")
prepend_path("CPATH","/opt/nvidia/hpc_sdk/Linux_x86_64/21.11/cuda/11.5/include")
prepend_path("CPATH","/opt/nvidia/hpc_sdk/Linux_x86_64/21.11/math_libs/11.5/include")
```

However, I was always manually pointing the compiler to `cuda_runtime.h` in pre- and post-AT, just the environment variable directory prefix changed:
```
-CUDACFLAGS=-I${CUDA_INSTALL_PATH}/include
+CUDACFLAGS=-I${NVIDIA_PATH}/cuda/include
```
See https://github.com/NVIDIA-developer-blog/code-samples/blob/master/posts/cuda-aware-mpi-example/src/Makefile

## pre-AT CUDA Toolkit just pointed to Cray NVHPC installations
In `/lus/swift/pAT/soft/modulefiles/cudatoolkit/`
### `11.6` Tcl Modulefile
```
#%Module

#
# cudatoolkit/11.6
#
# modulefile template for CTK 11.6
#
# sample path for this file: /opt/nvidia/modulefiles/cudatoolkit/11.6
#
# modify PKG_CONFIG_PATH with dir with corresponding 'ctk-11.6.pc' pkg-config file
#
# modify cu_base cu_math cu_nvvm cu_cupt to match CTK install paths
#

conflict "cudatoolkit"

set cu_prefix "/opt/nvidia/hpc_sdk/Linux_x86_64/22.3"

# check these paths in the CTK SDK install
set cu_base "$cu_prefix/cuda/11.6"
set cu_math "$cu_prefix/math_libs/11.6"
set cu_nvvm "$cu_prefix/cuda/11.6/nvvm"
set cu_cupt "$cu_prefix/cuda/11.6/extras/CUPTI"

setenv cudatoolkit_VERSION "11.6"

setenv CRAY_CUDATOOLKIT_VERSION "11.6"

#
# we add "ctk-x.x" to PE_PKGCONFIG_LIBS to match our corresponding "ctk-x.x.pc" file prefix
#

prepend-path PE_PKGCONFIG_LIBS "ctk-11.6"

#
# we modify PKG_CONFIG_PATH with dir with corresponding ctk-11.4.pc pkg-config file
#

prepend-path PKG_CONFIG_PATH "/soft/modulefiles/cudatoolkit"

prepend-path LD_LIBRARY_PATH "$cu_base/lib64:$cu_math/lib64:$cu_cupt/lib64:$cu_nvvm/lib64"
prepend-path PATH "$cu_base/bin"
```

### `ctk-11.6.pc`

```
Name: cudatoolkit
Version: 11.6
Description: NVIDIA cudatoolkit

#
# ctk-11.6.pc
#
# pkg-config file template for CTK 11.6
#
# works alongside the cudatoolkit/11.6 environment modulefile
#
# modify cu_base cu_math cu_nvvm cu_cupt to match CTK install paths
#

cu_base=/opt/nvidia/hpc_sdk/Linux_x86_64/22.3/cuda/11.6
cu_math=/opt/nvidia/hpc_sdk/Linux_x86_64/22.3/math_libs/11.6
cu_nvvm=/opt/nvidia/hpc_sdk/Linux_x86_64/22.3/cuda/11.6/nvvm
cu_cupt=/opt/nvidia/hpc_sdk/Linux_x86_64/22.3/cuda/11.6/extras/CUPTI

Cflags: -I${cu_base}/include -I${cu_cupt}/include -I${cu_nvvm}/include

Libs: -L${cu_base}/lib64 -L${cu_cupt}/lib64 -L${cu_nvvm}/lib64 -Wl,--as-needed,-lcupti,-lcudart,--no-as-needed -l\
cuda
```
