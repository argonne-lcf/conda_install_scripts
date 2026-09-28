# Fall 2026 (`conda/2026-10-01`) to-do and notes

`conda/2026-10-01` (Python 3.13, CUDA 13.0.3, torch 2.14.0, TF 2.21.0, JAX 0.11.1, vLLM 0.29.0,
DeepSpeed 0.19.7, TE 2.19) was built on Sirius 2026-09-24 with `build_monolithic_conda_module_resumable.sh`
(final job 32642, after 6 failed attempts; the build logs were not kept). Modulefile
`modulefiles/conda/2026-10-01.lua` and `.modulerc.lua` (hides the `conda/2026-09-17` prototype) are on
Sirius `/soft/modulefiles/conda/`. Not the default yet; `conda/2025-09-25` still is. Env is read-only
and world-readable (checked 2026-09-24). All conda modules older than the September 2025 builds were deleted.

**Quick package adds for the next build**
- [ ] `torchtitan` 0.3.0 (PyPI, 2026-09-03; Python >=3.11). Not in 2026-10-01. Its deps are already in
  the env except `torch_checkpointing`, `tyro`, `spmd_types==0.2.3`. It caps `datasets<4.8.0` vs 5.0.1
  in 2026-10-01: install `--no-deps` and check its HF dataloader (leave the cap violated, like click for
  globus-compute). verl's TorchTitan engine (`model_engine=torchtitan`) is separate and harder: it wants
  a torchtitan nightly plus torch `>=2.14.0.dev20260625` with the `spmd_types` backend (verl
  `docs/workers/torchtitan_workers.rst`); untested against our torch 2.14.0 release + vLLM 0.29.0.
- [x] TRL and `ray[rllib]` (pinned to the installed ray): in both build scripts since 38e5e39; not in 2026-10-01.
- TorchRL (needs tensordict 0.14, which breaks verl) and OpenRLHF (cp310-cp312 wheels only, hard pins)
  deliberately left out; see the build script comments.

**To do**
- [ ] Sync to Polaris: `/soft/applications/conda/2026-10-01/`, `/soft/modulefiles/conda/2026-10-01.lua`,
  `/soft/modulefiles/conda/.modulerc.lua` (decide on `/soft/applications/conda/pkgs/`). The source trees
  `pytorch/` (11G), `vllm/` (9.6G), `tensorflow/` (1.9G), `bazel-7.7.0/` are not needed at runtime (no
  editable installs), ~22G of the 45G.
- [x] On Polaris: rerun the 2-node harness and the per-package isolation tests
  (`tests/polaris-2026-10-01.pbs`, job 7651437; see Verification (Polaris) below).
- [x] On Polaris: `tests/perf/perf-2026-10-01.pbs` (job 7651564; see Verification (Polaris) below).
- [ ] Slack announcement to the testing group; polaris-users listserv items (available now / default
  change); docs PR in `user-guides` (system-updates entries, framework pages off `conda/2024-04-29`).
- [ ] Change the default in `.modulerc.lua` after ~2 weeks of testing.
- [ ] `conda/2026-10-01-aws-nccl-<ver>` variant: aws-ofi-nccl against NCCL 2.30 / CUDA 13. Until then
  cross-node NCCL runs over TCP sockets (2-node DDP/FSDP slower than 1 node except HSDP).
- [ ] Next build: uninstall the CUDA 13.4 runtime/compiler wheels that vLLM's `humming-kernels[cu13]`
  pulls in, and re-test (see below): `pip uninstall -y nvidia-cuda-runtime nvidia-cuda-nvrtc
  nvidia-cuda-nvcc nvidia-cuda-crt nvidia-nvvm nvidia-cuda-cccl`. Keep `nvidia-nvshmem-cu13` (torch's
  only NVSHMEM), `nvidia-cutlass-dsl`, `nvidia-ml-py`.
- [ ] Next build: `cuda-bindings==13.0.3` pin does not hold; `cuda-python` 13.4.1 (unbounded dep of
  flashinfer-python / nvshmem4py-cu13) requires `cuda-bindings~=13.4.1`, env ended at 13.4.3.
- [ ] `/soft/modulefiles/jax/0.4.26`, `0.4.29-dev` are stale (pre-upgrade); docs no longer point there.
- [ ] Admin pip installs into the read-only env: use `--no-user`, else pip silently falls back to a user
  install (with the module unloaded, into `~/.local/lib/python3.13`, invisible under the module's
  `PYTHONUSERBASE`). Happened with `multipledispatch` on 2026-09-24.
- [ ] **Known bug, shipped in `conda/2026-10-01` (not patched in place; fixed in both build scripts
  for the next build):** megatron-core does not import. `import megatron.core.models.gpt` (and
  verl's Megatron engine) raises `AttributeError: module 'cutlass.cute.core' has no attribute
  'ThrMma'`: flash-attn 2.8.3's bundled `flash_attn/cute` FA4 prototype predates nvidia-cutlass-dsl
  4.8, and megatron-core 0.19.2's `attention.py` imports it but only catches `ImportError`. Fix:
  delete `site-packages/flash_attn/cute` (unused without the separate `flash-attn-4` dist).
- [ ] **Known bug, shipped in `conda/2026-10-01` and `2026-09-17` (same status):** verl's trainer
  entry points (`main_ppo`, `ray_trainer`, `sft_trainer`, `engine_workers`, vLLM rollout) fail with
  `No module named 'orjson'`: `verl/utils/tracking.py` imports it, verl does not declare it, and we
  install verl `--no-deps`. Fix: `pip install --no-deps orjson`. Workaround for users now: a
  `--system-site-packages` venv (or `pip install --user orjson`); for megatron,
  `import sys; sys.modules["flash_attn.cute"] = None` before importing megatron (how the fix was tested).
  Verified both fixes on a compute node (2026-09-25, `~/verl-td-test/verify_fix.out`); the
  isolation tests now have `mcore`, `verl` and `rl` cases so the harness catches this.

**Verification (Sirius, 2026-09-24)**
- 2-node harness (job 32648): all checks OK on 8 ranks, same as 09-17. Isolation tests (job 32649):
  all OK; numpyro OK after `multipledispatch` was added (build script now installs it).
- Perf regression, synthetic, same 2 Sirius nodes (job 32650, `tests/perf/perf-sirius-2026-10-01.{pbs,out}`,
  logs in `tests/perf/logs/32650/`). No regression; 8-GPU DDP difference is within cross-node TCP noise.

  | test | 2026-10-01 | 2026-09-17 |
  |---|---|---|
  | DDP ResNet-50, 1 GPU | 1880 img/s | 1877 |
  | DDP ResNet-50, 4 GPUs | 7157 img/s | 7153 |
  | DDP ResNet-50, 8 GPUs (2 nodes) | 5584 img/s | 5824 |
  | FSDP2 1.42B, 4 GPUs | 72.7k tok/s, 166 TFLOPS/GPU | 72.5k, 165 |
  | FSDP2 1.42B, 8 GPUs full shard | 5.26k tok/s | 5.25k |
  | FSDP2 1.42B, 8 GPUs HSDP (2x4) | 11.8k tok/s | 11.8k |

**Verification (Polaris, 2026-09-24)**
- `tests/polaris-2026-10-01.{pbs,out}` (job 7651437): 2-node harness exit 0, all 14 isolation tests
  rc=0 (incl. numpyro).
- Perf (job 7651564, `tests/perf/perf-2026-10-01.{pbs,out}`, logs in `tests/perf/logs/7651564/`), same
  two nodes as the 2026-09-17 Polaris run (job 7646687). Single-node numbers within 0.3%; the 2-node
  gains are TCP run-to-run noise (Sirius showed the same spread), not an improvement. The DDP 4-GPU
  and HSDP RESULT lines are garbled by NCCL INFO output in the .out; numbers here are from the logs.

  | test | 2026-10-01 | 2026-09-17 |
  |---|---|---|
  | DDP ResNet-50, 1 GPU | 1878 img/s | 1880 |
  | DDP ResNet-50, 4 GPUs | 7164 img/s | 7159 |
  | DDP ResNet-50, 8 GPUs (2 nodes) | 6092 img/s | 5633 |
  | FSDP2 1.42B, 4 GPUs | 72.9k tok/s, 166 TFLOPS/GPU | 73.1k, 166 |
  | FSDP2 1.42B, 8 GPUs full shard | 5.14k tok/s | 5.12k |
  | FSDP2 1.42B, 8 GPUs HSDP (2x4) | 12.4k tok/s | 11.6k |
  | vLLM Qwen2.5-7B TP=4, 1 node (mp) | 22.9k gen tok/s, init 62 s | 22.9k, init 104 s |
  | vLLM Qwen2.5-7B TP=4 PP=2, 2 nodes (ray) | 6.3k gen tok/s, init 224 s | 5.0k, init 251 s |

**NVIDIA pip wheels vs /soft CUDA** (`tests/libprobe.py`, `tests/libprobe-2026-10-01.{pbs,out}`, job 32651)
- vLLM's requirements (`humming-kernels[cu13]`) install `nvidia-cuda-{runtime,nvrtc,nvcc,crt,cccl}`
  13.4.92 and `nvidia-nvvm`; FlashInfer/vLLM pull `nvidia-cutlass-dsl`; `nvshmem4py-cu13` pulls
  `nvidia-nvshmem-cu13`. Same set in `conda/2026-09-17`.
- They are used, not just installed: torch 2.14's `_load_global_deps()` calls `_preload_cuda_deps(required=False)`
  so any component wheels present win the soname lookups, and JAX's `xla_cuda13` plugin "prefer[s] the
  Python packages, if present". Every probed process (torch, JAX, CuPy, TE, FlashInfer, vLLM) maps
  wheel `libcudart.so.13` (13.4) and `libnvrtc` 13.4 alongside /soft 13.0.3 cudart; cuBLAS, cuDNN, NCCL,
  nvJitLink come from /soft. All probes pass on driver 580 (CUDA 13.0), incl. torch jiterator (NVRTC)
  and a CuPy ElementwiseKernel: NVRTC emits sm_80 cubins. Risk is only code that hands the driver PTX
  newer than 13.0. Not a deploy blocker; fix in the next build (to-do above).

**Most troublesome packages to build (2026-09-17 + 2026-10-01 rounds)**
1. vLLM: `use_existing_torch.py` plus trimming `requirements/cuda.txt`; its resolver silently reinstalled
   numpy/numba/tilelang (hence the frozen constraints file); exact FlashInfer pin; `ninja -j2` crawl
   with the old MAX_JOBS; setuptools `<81` cap vs 84 (attempt 4); `import vllm` from its source tree
   (attempt 6); source of the CUDA 13.4 wheels; 9.6G source tree.
2. TensorFlow: hermetic bazel + clang 22; lock files only to Python 3.13 (pins the whole env); its
   RTLD_GLOBAL LLVM crashed triton (hook file deleted); import from source dir (attempt 1); old C++ ABI.
3. DeepSpeed: CUDA 13 detection picked C++17 (fixed upstream in 0.19.7); `fp_quantizer` fails under
   nvcc 13 (`DS_BUILD_FP_QUANTIZER=0`, attempt 2); NCCL include path; `deepspeed-kernels` pulled a PyPI
   cmake that clobbered bin/cmake.
4. TransformerEngine: cudnn-frontend, `NCCL_HOME`, `CUDNN_PATH`, `NVTE_WITH_NCCL_EP=0` on A100,
   `-DNCCL_INCLUDE_DIR`; <= 2.7 needed `NVTE_CUDA_INCLUDE_DIR` at runtime.
5. flash-attn / mamba-ssm / causal-conv1d: sdists hard-code `-std=c++17` vs torch's C++20
   (`pip_sdist_cxx20`); force-build flags; tilelang/apache-tvm-ffi pins permanently conflict with vLLM's.
6. globus-compute + parsl: locked to the Ops endpoint versions; standing dill/click/psutil conflicts.
7. xgboost: scikit-build-core migration; isolated build fetched cmake 4 (`--no-build-isolation`).
8. verl: pure Python, installs fast; the pain is pins (`transformers<5.11`, 0.9.1 `requires-python<3.13`,
   downgrades `packaging` to 25.0).

Also: FlashInfer from source (slower than vLLM; now pinned wheels), PyG extensions (torch_scatter
discontinued), h5py/mpi4py/mpi4jax GTL linking (`craype-accel-nvidia80` workaround).
Resume-only pitfall (resumable script): re-running the conda install re-links conda's setuptools
egg-info next to pip's; the script now uninstalls every copy before reinstalling (attempt 5).

# Fall 2025 to-do and notes

`conda/2025-09-25` was the first draft module on Sirius to make it through deployment to Polaris in this round.
- Built with vLLM `0.11.0rc2.dev147+g47b933954.d20251006.cu129`, which was incompatible with Verl
- **Sirius and Polaris copies are currently out of sync**. I experimented with post-build fixes to vLLM and Verl on Sirius, resulting in vLLM `0.9.2.dev0+gb6553be1b.d20251008.cu129` on Sirius, which mostly worked. **Requested that Ops overwrite the Sirius `/soft/applications/conda/2025-09-25/` with the contents of the Polaris one**
  - Monitor the progress of the reverse sync in the afternoon of 2025-10-24

~~On only Sirius right now / never synced to Polaris:~~
- `conda/2025-09-26` is a v2, mostly debugging Verl, but also some modulefile minor improvements related to JITing
- `conda/2025-09-27` was just a complete end-to-end, fresh build from the build script, no post-script changes.
  - Not really used; later deleted from Sirius completely. 
- `conda/2025-09-28` full draft v3, with `flash-attn` pinned to 2.8.2

**2026-09-15 post-hoc update:** turns out, I did have 3x `conda/2025-09-28*` intentionally synced over to Polaris on 2025-10-27 as a final version of this pass of phase of conda releases. It appears I accidentally synced the 3x `conda/2025-09-26*` modules over to Polaris. Meant to sync small QoL changes to the 3x default `conda/2025-09-25*`. Asked A. Meyer to remove them on 2025-10-24, but it appears that this never occurred. So now we have the following modulefiles on Polaris (and still in sync with Sirius):
```console
❯ ll
total 116K
-rw-r--r-- 1 felker software 5.7K Nov 19  2025 2024-04-29-aws-nccl.lua
-rw-r--r-- 1 felker software 4.8K Nov 19  2025 2024-04-29.lua
-rw-r--r-- 1 felker software 4.7K Nov 19  2025 2024-10-30-workshop.lua
-rw-r--r-- 1 felker software 5.4K Nov 19  2025 2025-09-25-aws-nccl-1.6.0.lua
-rw-r--r-- 1 felker software 5.4K Nov 19  2025 2025-09-25-aws-nccl-1.9.1.lua
-rw-r--r-- 1 felker software 4.9K Nov 19  2025 2025-09-25.lua
-rw-r--r-- 1 felker software 5.7K Nov 19  2025 2025-09-26-aws-nccl-1.6.0.lua
-rw-r--r-- 1 felker software 5.7K Nov 19  2025 2025-09-26-aws-nccl-1.9.1.lua
-rw-r--r-- 1 felker software 4.9K Nov 19  2025 2025-09-26.lua
-rw-r--r-- 1 felker software 4.9K Nov 17  2025 2025-09-26-xalt.lua
-rw-r--r-- 1 felker software 5.7K Nov 19  2025 2025-09-28-aws-nccl-1.6.0.lua
-rw-r--r-- 1 felker software 5.7K Nov 19  2025 2025-09-28-aws-nccl-1.9.1.lua
-rw-r--r-- 1 felker software 4.9K Nov 19  2025 2025-09-28.lua
-rw-r--r-- 1 felker software 4.9K Nov 17  2025 2025-09-28-xalt.lua
-rw-r--r-- 1 felker software  188 Oct 24  2025 .modulerc.lua
```
The only differences among the dated 3x sets of modulefiles are:
```lua
local base_path = "/soft/applications/conda/2025-09-25/"
```
And `2025-09-26-xalt.lua / 2025-09-28-xalt.lua ` can both be deleted. Identical to their non-suffixed versions. 

**To Do (late October)**:
- [x] `xformers` (`0.0.32.post2`) and `flash-attn` (`2.8.3`) version mismatch in both `conda/2025-09-25` (on both Sirius and Polaris) and `conda/2025-09-26` (Filippo):
```console
$ module use /soft/modulefiles/ && module load conda/2025-09-25 && conda activate
$ python -c "from xformers.ops.fmha import flash"
Traceback (most recent call last):
  File "<string>", line 1, in <module>
  File "/soft/applications/conda/2025-09-25/mconda3/lib/python3.12/site-packages/xformers/ops/__init__.py", line 9, in <module>
    from .fmha import (
  File "/soft/applications/conda/2025-09-25/mconda3/lib/python3.12/site-packages/xformers/ops/fmha/__init__.py", line 10, in <module>
    from . import attn_bias, ck, ck_splitk, cutlass, flash, flash3, triton_splitk
  File "/soft/applications/conda/2025-09-25/mconda3/lib/python3.12/site-packages/xformers/ops/fmha/flash.py", line 82, in <module>
    raise ImportError(
ImportError: Requires Flash-Attention version >=2.7.1,<=2.8.2 but got 2.8.3.
```
- [ ] Leftover `verl_test-0.5.0-vllm0.9.1.sh` Verl test script errors on Sirius--- compute node specific errors?? Keeps failing due to thread/process limits on node `x3200c0s13b0n0`, but seems to work on other nodes: `x3200c0s13b1n0, x3200c0s19b1n0, x3200c0s1b0n0`. See messages in `#alcf_sirius` and with Adam on 2025-10-09. Adam sees: `2025-10-09T20:32:47.459973+00:00 x3200c0s13b0n0 kernel: [T3221845] cgroup: fork rejected by pids controller in /jobs/21602`
  - Adam confirmed that there was something wrong with `x3200c0s13b0n0`. Was added to the `build` node set originally, then removed and possibly rebuilt incorrectly. Leading to messed up cgroups? Not sure of the specifics. I always getting this node with `build=false` in my `qsub` select list--- was the test script ever used with `build=true`?
- [ ] Something changed on Polaris compute node interactive jobs: they don't seem to launch Zsh login shells, or maybe don't inherit the environment variables from the parent shell on the login node. So `/home/felker/bin:/home/felker/mygit/bin:/home/felker/myemacs` etc. werent appearing in my `PATH`, and `ssh` commands are failing. **Current workaround:** sourcing `~/.env` in `~/.zshrc` instead of `~/.zlogin` ---> suboptimal, because nested interactive shells result in duplicate `PATH` entries
- [ ] Probably related: building `flash-attn` from source in a venv on Polaris compute node keeps erroring out (around [~25/75\] Ninja tasks) due to `HTTPError`. No such issue on Sirius compute nodes:
```
module use /soft/modulefiles; module load conda/2025-09-25; conda activate base
CONDA_NAME=$(echo ${CONDA_PREFIX} | tr '\/' '\t' | sed -E 's/mconda3|\/base//g' | awk '{print $NF}')
VENV_DIR="$(pwd)/venvs/${CONDA_NAME}"
mkdir -p "${VENV_DIR}"
python -m venv "${VENV_DIR}" --system-site-packages
source "${VENV_DIR}/bin/activate"

export DISABLE_PYMODULE_LOG=1

pip install --no-build-isolation "flash-attn==2.8.2"
```
results in 
```
      Traceback (most recent call last):                                                        05:27:53 [89/3636]
        File "/var/tmp/pbs.6558019.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov/pip-install-r5_e2cq3/flash-attn_31b6
26704caf4534aff4533831d1ae07/setup.py", line 485, in run
          urllib.request.urlretrieve(wheel_url, wheel_filename)
        File "/soft/applications/conda/2025-09-25/mconda3/lib/python3.12/urllib/request.py", line 240, in urlretri
eve
          with contextlib.closing(urlopen(url, data)) as fp:
                                  ^^^^^^^^^^^^^^^^^^
        File "/soft/applications/conda/2025-09-25/mconda3/lib/python3.12/urllib/request.py", line 215, in urlopen
          return opener.open(url, data, timeout)
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        File "/soft/applications/conda/2025-09-25/mconda3/lib/python3.12/urllib/request.py", line 521, in open
          response = meth(req, response)
                     ^^^^^^^^^^^^^^^^^^^
        File "/soft/applications/conda/2025-09-25/mconda3/lib/python3.12/urllib/request.py", line 630, in http_res
ponse
          response = self.parent.error(
                     ^^^^^^^^^^^^^^^^^^
        File "/soft/applications/conda/2025-09-25/mconda3/lib/python3.12/urllib/request.py", line 559, in error
          return self._call_chain(*args)
                 ^^^^^^^^^^^^^^^^^^^^^^^
        File "/soft/applications/conda/2025-09-25/mconda3/lib/python3.12/urllib/request.py", line 492, in _call_ch
ain
          result = func(*args)
                   ^^^^^^^^^^^
        File "/soft/applications/conda/2025-09-25/mconda3/lib/python3.12/urllib/request.py", line 639, in http_err
or_default
          raise HTTPError(req.full_url, code, msg, hdrs, fp)
      urllib.error.HTTPError: HTTP Error 404: Not Found

      During handling of the above exception, another exception occurred:

      Traceback (most recent call last):
        File "/soft/applications/conda/2025-09-25/mconda3/lib/python3.12/site-packages/torch/utils/cpp_extension.p
y", line 2595, in _run_ninja_build
          subprocess.run(
        File "/soft/applications/conda/2025-09-25/mconda3/lib/python3.12/subprocess.py", line 571, in run
          raise CalledProcessError(retcode, process.args,
      subprocess.CalledProcessError: Command '['ninja', '-v', '-j', '32']' returned non-zero exit status 255.
```

- [ ] (unrelateD) ALCF User Guides documentation updates requested by Venkat for [PyTorch on **Aurora** page](https://docs.alcf.anl.gov/aurora/data-science/frameworks/pytorch/):
  - [ ] Add Major Changes section on version 2.5 to 2.8 **at the bottom**. Focus on users who are not familiar with Intel XPUs at the top
  - [ ] Update [vLLM on Aurora page too](https://docs.alcf.anl.gov/aurora/data-science/inference/vllm/)
- [x] Khalid confirms `conda/2025-09-25` operations using `conda` throw at least 5x stale `conda-meta` jsons warnings. I can reproduce on a Polarislogin node, just run `conda list`. **Manually delete the files, for now**
```
(2025-09-25/base) hossainm@x3106c0s37b0n0:~> conda list | grep "datasets"                               
WARNING conda.gateways.disk.delete:unlink_or_rename_to_trash(183): Could not remove or rename /soft/applications/conda/2025-09-25/mconda3/conda-meta/numpy-2.3.3-py312h33ff503_0.json.  Please remove this file manually (you may need to reboot to free file handles)
```
also `numba-0.62.1-py312h907b442_0.json`, `setuptools-80.9.0-pyhff2d567_0.json`, `cffi-1.17.1-py312h06ac9bb_0.json`, `llvmlite-0.45.1-py312h7424e68_0.json`

- [x] Upstream modulefile improvements from later versions to `conda/2025-09-25`'s modulefile:
```diff
❯ diff 2025-09-25.lua 2025-09-26.lua
31a32,36
> depends_on("gcc-native/14.2")
> -- note, unloading this does not remove /usr/bin/g++-14, e.g.
> -- Just means /usr/bin/g++ (7.5.0) is the first in path, not /opt/cray/pe/gcc-native/14/bin/g++
> -- TODO: replace /usr/bin/g++-14 etc. in build script with /opt/cray/pe/gcc-native/14/bin/g++
> -- are they identical??
33c38,47
< local conda_dir = "/soft/applications/conda/2025-09-25/mconda3"
---
> -- helps when vLLM JIT compiles things
> setenv("CC","/usr/bin/gcc-14")
> setenv("CXX","/usr/bin/g++-14")
>
> setenv("TORCH_CUDA_ARCH_LIST","8.0")
> setenv("FLASHINFER_CUDA_ARCH_LIST","8.0")
>
> local base_path = "/soft/applications/conda/2025-09-26/"
> setenv("BASE_PATH",base_path)
> local conda_dir = pathJoin(base_path,"mconda3")
```

- [x] Bug in `sitecustomize.py` on Polaris causing issues with PyTorch (Sam). **Applying his patch** in https://github.com/argonne-lcf/PyModuleSnooper/commit/b23719a0867905f54faca11d2d6960b814ab9263
```console
(2025-09-25/base) [15:50 bicer@x3101c0s13b1n0 polaris]$ python
Python 3.12.11 | packaged by conda-forge | (main, Jun 4 2025, 14:45:31) [GCC 13.3.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import torch
>>> torch.__version__
'2.8.0'
>>> quit()
Exception ignored in atexit callback: <function inspect_and_log at 0x14c523080540>
Traceback (most recent call last):
File "/soft/applications/conda/2025-09-25/mconda3/lib/python3.12/site-packages/sitecustomize.py", line 106, in inspect_and_log
module_name: str(getattr(module, "__version__", None))
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
File "<frozen importlib._bootstrap>", line 552, in _module_repr
File "/soft/applications/conda/2025-09-25/mconda3/lib/python3.12/site-packages/torch/_classes.py", line 13, in __getattr__
proxy = torch._C._get_custom_class_python_wrapper(self.name, attr)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
RuntimeError: Tried to instantiate class '__version__.__file__', but it does not exist! Ensure that it is registered via torch::class_
```

**To Do (early October)**:

[My Slack post summarizing the v1 draft of this module on 2025-10-02](https://cels-anl.slack.com/archives/C3FU1QXHR/p1759434706730799)
- [ ] PyTorch's DLPack extension compilation? Need to force it to use `CC=/usr/bin/gcc-14 CXX=/usr/bin/g++-14` when compiling. Otherwise it fails
```
/soft/applications/conda/2025-09-25/mconda3/lib/python3.12/site-packages/tvm_ffi/_optional_torch_c_dlpack.py: ...
---> $HOME/.cache/torch_extensions/py312_cu129/c_dlpack/main.cpp
```
export them in the modulefile? This also happens in Verl test script--- hard to know when vLLM JITs some things

- [ ] `qstat` not available on Sirius compute node, unlike login nodes. Reported to Cyrus. Need to `export PATH=$PATH:/opt/pbs/bin`, for now. Needed for `ezpz-test`. **Currently adopting a workaround in my personal dotfiles, not in the modulefile**
- [ ] Hope Ops increases per-user cgroups process limit from 128 to 512 or 1024
- [ ] Evaluate if this is a bug: vLLM initialization and subsequent calls must be wrapped in a `if __name__ == '__main__':` block. This ensures that the code that spawns new processes is only executed once in the parent process.
- [ ] Test AWS v1.9.1 plugin, once HPE says that it is validated and recommended to use
- [x] Update ezpz to https://github.com/saforem2/ezpz/tree/saforem2/tests
- [x] Update ezpz again to v0.9.0
- [x] Confirm that build script `build_monolithic_conda_module.sh` runs completely, first to last line, without error or need for manual intervention and fixes. **yes, as of 5bef5c90cb3**
- [x] Get green light to deploy, and check language on ALCF Updates email.
- [x] Notify Ops ALCF Sirius Slack channel to sync
- [x] Email ALCF media. Also add to `polaris-notify`
- [x] Add to https://docs.alcf.anl.gov/polaris/system-updates/
- [ ] Change `.modulerc.lua` default in two weeks (announce beforehand)
- [x] Even just running `mpi4py` on two Sirius compute nodes within `ipython`: there is an issue due to missing filesystem mount
```output
darshan_library_warning: unable to create log file /lus/grand/logs/darshan/polaris/2025/10/6/felker_python3.12_id21511-1500633_10-6-4864-8027967647074095540.darshan_partial.
```
Fix:
```
# Darshan fails on Sirius because Grand is not mounted
❯ module unload darshan
❯ export DARSHAN_DISABLE=1
```
- [ ] **Someday**: find a workaround to NFS write/read/permission errors `~/.cache` etc. also `/home/felker/.config/matplotlib/stylelib/ambivalent` during `ezpz-test`
- [ ] Port build script to ALCF Sophia
- [ ] Consider exposing the following libraries (used in the build script) in the modulefile in some capacity:
  - [ ] NVSHMEM 3.3.9
  - [ ] `CUTLASS_PATH` (I used the latest main branch; what version?)
  - [ ] `BASE_PATH`
  - [ ] cuSPARSELt 0.8.11
  - [ ] Add more version info to the modulefile's `help()`? SGLang, vLLM, Verl, DS, etc. versions
- [x] Clean up the micro PyTorch environments (`2025-10-05-pt`, `2025-10-05-pt-v2`) and old modulefiles
- [ ] **Explain**: why didnt the following modulefile hotfix work with the original PyTorch build. Or did it, and I just didnt understand the below nuances and limitations?
```lua
prepend_path("LD_PRELOAD", pathJoin(os.getenv("CRAY_MPICH_DIR") or "/opt/cray/pe", "lib/libmpi_gtl_cuda.so"))
```
- [ ] Fix this: (might have happened after some last minute manual changes to the build?)
```console
❯ conda list
WARNING conda.gateways.disk.delete:unlink_or_rename_to_trash(183): Could not remove or rename /soft/applications/c
onda/2025-09-25/mconda3/conda-meta/setuptools-80.9.0-pyhff2d567_0.json.  Please remove this file manually (you may need to reboot to free file handles)
WARNING conda.gateways.disk.delete:unlink_or_rename_to_trash(183): Could not remove or rename /soft/applications/conda/2025-09-25/mconda3/conda-meta/numpy-2.3.3-py312h33ff503_0.json.  Please remove this file manually (you may ne
ed to reboot to free file handles)
WARNING conda.gateways.disk.delete:unlink_or_rename_to_trash(183): Could not remove or rename /soft/applications/c
onda/2025-09-25/mconda3/conda-meta/numba-0.62.1-py312h907b442_0.json.  Please remove this file manually (you may n
eed to reboot to free file handles)
WARNING conda.gateways.disk.delete:unlink_or_rename_to_trash(183): Could not remove or rename /soft/applications/c
onda/2025-09-25/mconda3/conda-meta/cffi-1.17.1-py312h06ac9bb_0.json.  Please remove this file manually (you may ne
ed to reboot to free file handles)
WARNING conda.gateways.disk.delete:unlink_or_rename_to_trash(183): Could not remove or rename /soft/applications/c
onda/2025-09-25/mconda3/conda-meta/llvmlite-0.45.1-py312h7424e68_0.json.  Please remove this file manually (you ma
y need to reboot to free file handles)
# packages in environment at /soft/applications/conda/2025-09-25/mconda3:
```

### Fix for mpi4py and PyTorch incompatibility
```bash
export USE_MPI=1
BUILD_TEST=0 CUDAHOSTCXX=g++-14 CC=cc CXX=CC LDFLAGS="-L/opt/cray/pe/lib64 -Wl,-rpath,/opt/cray/pe/lib64 -lmpi_gtl_cuda ${LDFLAGS}" python setup.py bdist_wheel 
```

- I am not sure what (if anything) in the module is broken if you try setting `export MPICH_GPU_SUPPORT_ENABLED=0`, but I am not particularly interested in supporting that use case (e.g. we set it `=1` by default in the modulefile)
- mpi4py and PyTorch Distributed with MPICH still seem to work, but you might get a performance hit relative to a module built entirely on the non-CUDA aware Cray libraries? The GTL libraries are hard-coded into the linker and loader via rpath. 
- The plugin’s library is guaranteed to be present regardless of the runtime setting, but I am not sure if MPICH disables the GPU-aware path and avoids the related overhead entirely, in that case

#### References
PyTorch Distributed only supports CUDA-Aware MPI Ops through OpenMPI: https://github.com/pytorch/pytorch/blob/2883b5ab773daf5861d43ff0b65be49a441ab3f9/torch/csrc/distributed/c10d/ProcessGroupMPI.cpp#L49-L62

Note, `export MPIX_CUDA_AWARE_SUPPORT=1` is likely not enough to trick the PyTorch build, since then it runs `if (MPIX_Query_cuda_support() == 1)`, a function that does not exist in Cray MPICH. It is conditionally imported:
```c++
#if defined(OPEN_MPI) && OPEN_MPI
#include <mpi-ext.h> // Needed for CUDA-aware check
#endif
```

https://docs.pytorch.org/docs/stable/distributed.html
> MPI supports CUDA only if the implementation used to build PyTorch supports it.
> ...
> MPI is an optional backend that can **only be included if you build PyTorch from source**. (e.g. building PyTorch on a host that has MPI installed.)

- The PyTorch docs barely even mention CUDA-aware MPI? Needed to look at the source code to find that tidbit about OpenMPI being the only supported distribution. 
- [2019 ticket from OLCFL Summit](https://code.ornl.gov/summit/mldl-stack/pytorch/-/issues/1) about "CUDA Aware MPI with Pytorch"
- https://forums.developer.nvidia.com/t/request-for-pytorch-wheel-with-mpi-backend-on-jetson-orin/340949/11
- https://github.com/pytorch/pytorch/issues/97507
- [CELS Slack convo](https://cels-anl.slack.com/archives/C3FU1QXHR/p1759769651108389)

#### Original problem in April 2024 build
This fails:
```python
import torch
from mpi4py import MPI
comm = MPI.COMM_WORLD
print(f"I am {comm.rank} of {comm.size}")
```
This will not fail:
```python
from mpi4py import MPI
import torch
comm = MPI.COMM_WORLD
print(f"I am {comm.rank} of {comm.size}")
```
===============================

**TensorFlow Hermetic build info**:
- https://github.com/tensorflow/tensorflow/commit/d9071b91c4e550e6c984357158c3460346616db5
- https://github.com/tensorflow/tensorflow/commit/0d2b08d354daddfd7a2d0f91aae56dae01aa82bc
- https://github.com/tensorflow/tensorflow/blob/master/.bazelrc
- https://github.com/tensorflow/tensorflow/commit/5c289f5ba22711a296a216100cf9816c6077d85d
- https://github.com/tensorflow/tensorflow/commit/3f4b2fda6ffe7dfe03c1663ef37f54fc4432cc8b
- https://github.com/tensorflow/tensorflow/commit/9b5fa66dc65753059cda686b6a5a8f16143bc5e0
- https://github.com/tensorflow/tensorflow/issues/78846
- https://github.com/tensorflow/tensorflow/issues/62459
- https://github.com/jax-ml/jax/issues/23689
- https://github.com/tensorflow/tensorflow/issues/86405
- https://github.com/openxla/xla/issues/20915#issuecomment-2566744479
- https://github.com/openxla/xla/blob/main/docs/hermetic_cuda.md
- https://github.com/openxla/xla/issues/27528
- https://openxla.org/xla/hermetic_cuda
- https://github.com/google-ml-infra/rules_ml_toolchain/tree/main/gpu

## October 2023 to-do
- [ ] New CUDA Graph + PyTorch issues that did not occur in `2022-09-08` (Lusch)
```
RuntimeError: CUDA error: operation not permitted when stream is capturing
CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1.
Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.
```

- [x] `chmod -R u-w /soft/datascience/conda/2023-10-04`
- [x] Save build log to VCS
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
