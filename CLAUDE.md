# conda_install_scripts

Build scripts and Lmod modulefiles for the ALCF `conda` modules (PyTorch, TensorFlow,
JAX, vLLM, DeepSpeed, ...) built from source on Polaris and its test system Sirius.
Primary target: `alcf_polaris/`. Modulefiles live in `alcf_polaris/modulefiles/conda/`
and are hand-synced to `/soft/modulefiles/conda/`; keep the two identical.

## Machine facts (Sirius/Polaris, PE 26.03, since the 2026-08-19 HPCM upgrade)

- SLES 15 SP7, cray-mpich 9.1.0, NVIDIA driver 580.65.06 (CUDA 13.0 driver API).
- `gcc-native/14` is GCC 14.3.0; `/usr/bin/gcc-14` is also 14.3.0. `gcc-native/14.2` no longer exists.
- `cray-hdf5-parallel/1.14.3.9` (libs are `libhdf5_parallel_gnu.so.310`; `1.14.3.5` is gone).
- Cray MPICH libs dropped the gcc suffix: `libmpi_gnu.so.12`, not `libmpi_gnu_123.so.12`.
- `/opt/cray/pe/lib64/libmpi_gtl_cuda.so.0` links `libcudart.so.13`. It coexists fine with a
  CUDA 12.x runtime in the same process. CUDA 13.x toolkits are in `/soft/compilers/cudatoolkit/`.
- The previous PE (cray-mpich 8.1.x, PE 25.03) is gone entirely; no fallback modules.
- Older builds are kept alive via `/soft/applications/conda/<date>/pe-26.03-shim/` symlink dirs
  prepended to `LD_LIBRARY_PATH` in the modulefile. See the README there.

## Building rules (learned the hard way)

- `module load craype-accel-nvidia80` refuses to load under PrgEnv-gnu (HPE case 5367752190).
  Export these instead before any `cc`-linked build:
  ```
  export CRAY_ACCEL_TARGET=nvidia80 CRAY_ACCEL_VENDOR=nvidia CRAY_CPU_TARGET=x86-64
  export CRAYPE_LINK_TYPE=dynamic CRAY_TCMALLOC_MEMFS_FORCE=1
  ```
  Every extension that loads libmpi (mpi4py, h5py, mpi4jax, torch) must link the GTL. If a
  GTL-less lib is imported first, MPI_Init aborts with "GTL library is not linked" because the
  modulefile sets `MPICH_GPU_SUPPORT_ENABLED=1`. Check with `ldd <ext>.so | grep gtl`.
- `module unload darshan` before building, or `cc` links libdarshan into every extension and
  users get `darshan_library_warning` on every Python exit.
- Any `pip install` into a shared env must use `--no-deps`. Without it pip replaced the
  Cray-linked mpi4py with the PyPI manylinux wheel and bumped numpy past numba's ceiling.
- Pin transitive pure-Python deps whose majors move under torch/jax:
  triton must match what the torch release pairs with (torch 2.8 -> triton 3.4.0; 3.5.0
  removed `triton_key` and broke `torch.compile`). jax must be one mpi4jax supports
  (jax 0.8.0 removed `mlir.custom_call`). The 2025-09-28 build shipped both broken.
- TransformerEngine <= 2.7 needs `NVTE_CUDA_INCLUDE_DIR=$CUDA_HOME/include` set in the
  modulefile or `import transformer_engine` crashes on the `nvidia` namespace package.
- torch.distributed's MPI backend does not support CUDA tensors with Cray MPICH (PyTorch only
  checks for OpenMPI). NCCL is the supported path; mpi4py itself is CUDA-aware.
- Login nodes have an 8 GB / 8 core / 128 PID per-user limit. Do heavy compiles on a compute node.

## Testing a module

- Use a batch job, not interactive `qsub -I`; this tool cannot drive an interactive session.
  A reusable 2-node harness (torch CUDA, NCCL all-reduce, CUDA-aware mpi4py, parallel h5py,
  TF, JAX, flash-attn, TE, DeepSpeed, vLLM, torch.compile, mpi4jax) is in `alcf_polaris/tests/`.
  ```
  qsub -A datascience -q workq -l select=2:ncpus=64:ngpus=4,filesystems=home:tegu,walltime=00:20:00
  mpiexec -n 8 --ppn 4 ./gpu_wrap.sh python gpu_test.py
  ```
- Per-rank GPU selection must happen in a wrapper script (`CUDA_VISIBLE_DEVICES=$PALS_LOCAL_RANKID`)
  before Python starts. Setting it inside Python is too late: the GTL initializes the CUDA driver
  during MPI_Init, and NCCL then sees every rank on GPU 0.
- Import h5py before mpi4py in tests; that order is the regression check for GTL-less builds.
- Test files must live on a shared filesystem (home or tegu), not `/tmp`.

## Lmod

- `module` is only defined in login shells: run through `bash -lc '...'`. Never pipe
  `module load` into grep; the pipeline subshell discards the environment.
- Stale `module avail` output comes from the per-user spider cache: `rm ~/.cache/lmod/spiderT.*`
  or `module --ignore_cache avail`. There is no site-wide cache.
- In `.modulerc.lua`, hide with `hide_version("conda/<full-name>")`. `hide_modulefile()` takes an
  absolute path and silently does nothing with a bare version string.

## Workflow preferences

- The maintainer runs pip installs and permission changes on `/soft` themselves. Hand over the
  exact command block and expected output; do not run it.
- Point to full job logs rather than summarizing failures away; the maintainer reads them.
- Commit only when asked. Commit messages: `polaris: <what>` with a short why-paragraph.
