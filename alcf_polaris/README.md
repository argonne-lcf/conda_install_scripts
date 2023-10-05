# October 2023 to-do
- [ ] New CUDA Graph + PyTorch issues that did not occur in `2022-09-08` (Lusch)
```
RuntimeError: CUDA error: operation not permitted when stream is capturing
CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1.
Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.
```

- [x] `chmod -R u-w /soft/datascience/conda/2023-10-04`
- [ ] Save build log to VCS
- [x] Remove `/soft/datascience/conda/2023-10-02`
- [ ] Rebuild DeepSpeed after next stable release following 0.10.3 with CUTLASS and Evoformer precompiled op support
- [ ] Update `pip` passthrough options to setuptools once pip 23.3 comes out (we are using 23.2.1): https://github.com/pypa/pip/issues/11859
- [ ] Unpin CUTLASS from fork with my 2x patches now that 1x patch is merged upstream
- [ ] Make `conda/2023-10-04` public; send notice to ALCF Announcements
- [ ] Make `conda/2023-10-04` the default; send notice to ALCF Announcements
- [ ] Update documentation
- [x] Still some spurious errors with `pyg_lib` (and `torch_sparse`, but that is just because it loads `libpyg.so`) at runtime when importing:
```
Cell In[1], line 1
----> 1 import pyg_lib

File /soft/datascience/conda/2023-10-02/mconda3/lib/python3.10/site-packages/pyg_lib/__init__.py:39
     35     else:
     36         torch.ops.load_library(spec.origin)
---> 39 load_library('libpyg')
     42 def cuda_version() -> int:
     43     r"""Returns the CUDA version for which :obj:`pyg_lib` was compiled with.
     44
     45     Returns:
     46         (int): The CUDA version.
     47     """

File /soft/datascience/conda/2023-10-02/mconda3/lib/python3.10/site-packages/pyg_lib/__init__.py:36, in load_library(lib_name)
     34     warnings.warn(f"Could not find shared library '{lib_name}'")
     35 else:
---> 36     torch.ops.load_library(spec.origin)

File /soft/datascience/conda/2023-10-02/mconda3/lib/python3.10/site-packages/torch/_ops.py:643, in _Ops.load_library(self, path)
    638 path = _utils_internal.resolve_library_path(path)
    639 with dl_open_guard():
    640     # Import the shared library into the process, thus running its
    641     # static (global) initialization code in order to register custom
    642     # operators with the JIT.
--> 643     ctypes.CDLL(path)
    644 self.loaded_libraries.add(path)

File /soft/datascience/conda/2023-10-02/mconda3/lib/python3.10/ctypes/__init__.py:374, in CDLL.__init__(self, name, mode, handle, use_errno, use_last_error, winmode)
    371 self._FuncPtr = _FuncPtr
    373 if handle is None:
--> 374     self._handle = _dlopen(self._name, mode)
    375 else:
    376     self._handle = handle

OSError: libpython3.10.so.1.0: cannot open shared object file: No such file or directory
```
Running `export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/soft/datascience/conda/2023-10-02/mconda3/lib` of course avoids the error, but is less than ideal. 
Occurred in `conda/2023-10-02`, which first had PyG and its deps built incorrectly (from wheels). I then rebuilt all of the dependency packages from source.
Does not occur in `conda/2023-10-04`, which had a similar journey, but only needed to rebuild `torch_sparse`, `torch_scatter` from source, since they required `CFLAGS` to be set?

Is this error dependent on the order of installation or something? Is `LDFLAGS` an issue? It isnt set until after PyG installation in the script, but then is set for post-script interactive sessions via the modulefile:
```
❯ echo $LDFLAGS
-L/soft/datascience/conda/2023-10-02/mconda3/lib -Wl,--enable-new-dtags,-rpath,/soft/datascience/conda/2023-10-02/mconda3/lib
```

```
> ldd /soft/datascience/conda/2023-10-02/mconda3/lib/python3.10/site-packages/libpyg.so
...
libpython3.10.so.1.0 => not found
```
**Answer**: indeed, having `LDFLAGS` set as above during `pip install --verbose git+https://github.com/pyg-team/pyg-lib.git` results in a broken `libpyg.so`. Rebuilding after `unset LDFLAGS`, then re-exporting the env var is fine. **However, this emphasizes that it might be problematic to have the LDFLAGS setting in the modulefile**. When users load the Anaconda module, and try to build additional software from source (esp. in a cloned conda env), this may cause serious issues. 

- [ ] Add warning about `LDFLAGS` to docs?
- [ ] Add Ray
- [ ] Add Redis, Redis JSON, newer DeepHyper
- [ ] Track all my GitHub issues from last 2x months
- [ ] Build new module with PyTorch 2.1.0 (released 2023-10-04); latest built is 2.0.1. Is the ATen cuDNN issue fixed?
- [ ] XLA performance regression?
- [ ] Confirm that removing Conda Bash shell function `__add_sys_prefix_to_path` for 2023 modules doesnt have adverse side effects. Document when/which conda version it was removed
- [ ] Known problem: no support for DeepSpeed Sparse Attention with Triton 2.x, PyTorch 2.x, Python 3.10
- [x] Confirm fix to `pip list | grep torch` version string via `PYTORCH_BUILD_VERSION`
- [ ] Decide on separate venv/cloned conda for `Megatron-DeepSpeed`
  - [ ] How volatile is the main branch, and how important is it to have the cutting edge version installed in a module on Polaris?
- [ ] Can we relax Apex being pinned to `52e18c894223800cb611682dce27d88050edf1de` ? What are the build failures on `master` 58acf96? Should we stick to tags like `23.08`, even though `README.md` suggests building latest `master`?
- [ ] What specific Apex features does `Megatron-DeepSpeed` rely on? `MixedFusedRMSNorm,FusedAdam,FusedSGD,amp_C,fused_weight_gradient_mlp_cuda`, multi-tensor applier for efficiency reasons, etc. How many of those are truly necessary? E.g. `amp_C` should be deprecated and PyTorch mixed precision should be used. Can a PR be opened to get rid of it?
- [ ] S. Foreman reporting multiple ranks place on a single GPU with PyTorch DDP? Specific to `Megatron-DeepSpeed`? Wrong NCCL version too; should be 2.18.3
```
torch.distributed.DistBackendError: NCCL error in: /soft/datascience/conda/2023-09-29/pytorch/torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp:1275, internal error, NCCL version 2.14.3
ncclInternalError: Internal check failed.
Last error:
Duplicate GPU detected : rank 0 and rank 4 both on CUDA device 7000
[...]
torch.distributed.DistBackendError: NCCL error in: /soft/datascience/conda/2023-09-29/pytorch/torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp:1275, internal error, NCCL version 2.14.3
ncclInternalError: Internal check failed.
Last error:
Duplicate GPU detected : rank 7 and rank 3 both on CUDA device c7000
    work = default_pg.barrier(opts=opts)
    work = default_pg.barrier(opts=opts)
torch.distributed.DistBackendError: NCCL error in: /soft/datascience/conda/2023-09-29/pytorch/torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp:1275, internal error, NCCL version 2.14.3
```
- [x] C. Simpson reporting classic Conda JSON permissions issues (was only on hacked-together `conda/2023-09-29`, not `conda/2023-10-04`):
```
(base)csimpson@polaris-login-01:/eagle/datascience/csimpson/dragon_public> conda list
WARNING conda.gateways.disk.delete:unlink_or_rename_to_trash(188): Could not remove or rename /soft/datascience/conda/2023-09-29/mconda3/conda-meta/wheel-0.38.4-py310h06a4308_0.json.  Please remove this file manually (you may need to reboot to free file handles)
WARNING conda.gateways.disk.delete:unlink_or_rename_to_trash(188): Could not remove or rename /soft/datascience/conda/2023-09-29/mconda3/conda-meta/jinja2-3.1.2-py310h06a4308_0.json.  Please remove this file manually (you may need to reboot to free file handles)
WARNING conda.gateways.disk.delete:unlink_or_rename_to_trash(188): Could not remove or rename /soft/datascience/conda/2023-09-29/mconda3/conda-meta/packaging-23.0-py310h06a4308_0.json.  Please remove this file manually (you may need to reboot to free file handles)
WARNING conda.gateways.disk.delete:unlink_or_rename_to_trash(188): Could not remove or rename /soft/datascience/conda/2023-09-29/mconda3/conda-meta/llvmlite-0.40.0-py310he621ea3_0.json.  Please remove this file manually (you may need to reboot to free file handles)
WARNING conda.gateways.disk.delete:unlink_or_rename_to_trash(188): Could not remove or rename /soft/datascience/conda/2023-09-29/mconda3/conda-meta/pyyaml-6.0-py310h5eee18b_1.json.  Please remove this file manually (you may need to reboot to free file handles)
WARNING conda.gateways.disk.delete:unlink_or_rename_to_trash(188): Could not remove or rename /soft/datascience/conda/2023-09-29/mconda3/conda-meta/numba-0.57.1-py310h0f6aa51_0.json.  Please remove this file manually (you may need to reboot to free file handles)
WARNING conda.gateways.disk.delete:unlink_or_rename_to_trash(188): Could not remove or rename /soft/datascience/conda/2023-09-29/mconda3/conda-meta/psutil-5.9.0-py310h5eee18b_0.json.  Please remove this file manually (you may need to reboot to free file handles)
WARNING conda.gateways.disk.delete:unlink_or_rename_to_trash(188): Could not remove or rename /soft/datascience/conda/2023-09-29/mconda3/conda-meta/cffi-1.15.1-py310h5eee18b_3.json.  Please remove this file manually (you may need to reboot to free file handles)
```
- [ ] ThetaGPU follow-up: should I be periodically be [running](https://docs.conda.io/projects/conda/en/latest/commands/clean.html) `conda clean --all`  to deal with permissions issues with the shared package cache there?


## GitHub issues and other tickets

- [ ] https://github.com/pytorch/pytorch/issues/107389
- [ ] https://github.com/pyg-team/pytorch_geometric/issues/8128
- [ ] https://github.com/microsoft/DeepSpeed/issues/4422
- [ ] https://github.com/google/jax/issues/17831
- [ ] https://github.com/NVIDIA/cutlass/issues/1118
  - [ ] https://github.com/NVIDIA/cutlass/pull/1121#event-10532195900
- [ ] https://github.com/pypa/pip/issues/12010

# Old 2022 to-do
- [x] Migrate away from `/soft/datascience/conda/2022-07-19-login`
- [x] Monitor fix to `umask` being messed up on compute nodes. Should be 022, not 077
- [x] Confirm PyTorch DDP functionality and performance with Corey
- [x] Re-check NCCL performnace with Denis once the GPUDirect issues are resolved: https://cels-anl.slack.com/archives/C03G5PHHF7V/p1658946362840349
- [x] Get copies of old, pre-AT Tcl Cray PE modulefiles for reference. Not possible, since they are not cached in `/lus/swift/pAT/soft/modulefiles`, but are part of the compute image in `/opt/cray/pe/lmod/modulefiles/core/PrgEnv-nvidia` e.g. 
  - [x] Understand `nvidia_already_loaded` etc. from https://cels-anl.slack.com/archives/GN23NRCHM/p1653506105162759?thread_ts=1653504764.783399&cid=GN23NRCHM (set to 0 in deprecated `PrgEnv-nvidia` and 1 in `PrgEnv-nvhpc` in AT Tcl modulefiles; **removed after AT**)
- [x] Get longer-term fix for `PrgEnv-gnu` failure building TF from source:
```
/opt/cray/pe/gcc/11.2.0/bin/redirect: line 5: /opt/cray/pe/gcc/11.2.0/bin/../snos/bin/redirect: No such file or directory
```
**Workaround:** `export GCC_HOST_COMPILER_PATH=/opt/cray/pe/gcc/11.2.0/snos/bin/gcc`. See 9ce52ceb1f0397822c9f1f177c5154aeb852a962. TensorFlow was never actually using `PrgEnv-nvhpc`; it was always pulling `export GCC_HOST_COMPILER_PATH=$(which gcc)` which is by default `/usr/bin/gcc` 7.5.0. 

**Is it a bug or unintended/unsupported use by TensorFlow installer?** Probably the latter. TF installer automatically "dereferences" the soft link. See https://stackoverflow.com/questions/7665/how-to-resolve-symbolic-links-in-a-shell-script (`realpath`, `pwd -P`) **Should never call the `redirect` shell script directly, only via `gcc` name.** The directory from which you call that `gcc` soft link or `redirect` script doesnt matter, FYI.


```
❯ /opt/cray/pe/gcc/11.2.0/bin/gcc
gcc: fatal error: no input files
compilation terminated.

❯ /opt/cray/pe/gcc/11.2.0/bin/redirect
/opt/cray/pe/gcc/11.2.0/bin/redirect: line 5: /opt/cray/pe/gcc/11.2.0/bin/../snos/bin/redirect: No such file or
directory

❯ cd /opt/cray/pe/gcc/11.2.0/bin/

❯ ./redirect
./redirect: line 5: ./../snos/bin/redirect: No such file or directory

❯ ll /opt/cray/pe/gcc/11.2.0/bin/gcc
lrwxrwxrwx 1 root root 8 Aug 14  2021 /opt/cray/pe/gcc/11.2.0/bin/gcc -> redirect*
```

The problem is that `basename /opt/cray/pe/gcc/11.2.0/bin/redirect` returns `redirect`, which is obviously not in the `/opt/cray/pe/gcc/11.2.0//snos/bin/`.
```
❯ ls /opt/cray/pe/gcc/11.2.0/bin/redirect
#! /bin/sh

eval ${XTPE_SET-"set -ue"}

$(dirname $0)/../snos/bin/$(basename $0) "$@"
```

- [x] No clue what `XTPE_SET` is. Cray-specific (e.g. XTPE = XT3 to XT6 programming environment) but not set in `PrgEnv-gnu`. Seems to XY Jin that it is for debugging the shell scripts. You might set it like `XTPE_SET='set -eux -o pipefail'`
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
