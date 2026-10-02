# Which CUDA user-space libs does each framework actually map at runtime: /soft CUDA 13.0.3
# or the nvidia-* 13.4 pip wheels (site-packages/nvidia/cu13)?  python libprobe.py <case>
import os, re, sys
which = sys.argv[1]
def t_torch():
    import torch
    x = torch.randn(512, 512, device="cuda"); (x @ x).sum().item()
    torch.special.bessel_j0(x).sum().item()          # jiterator -> NVRTC
    torch.compile(lambda a: (a * 2).sin())(x).sum().item()   # triton
def t_jax():
    import jax, jax.numpy as jnp
    jax.jit(lambda a: jnp.sin(a) @ a)(jnp.ones((256, 256))).block_until_ready()
def t_cupy():
    import cupy as cp
    k = cp.ElementwiseKernel("float32 x", "float32 y", "y = sinf(x) * 2", "probe_k")
    k(cp.ones(1024, dtype=cp.float32)); cp.cuda.Device().synchronize()
def t_te():
    import torch, transformer_engine.pytorch as te
    te.Linear(64, 64).cuda()(torch.randn(8, 64, device="cuda")); torch.cuda.synchronize()
def t_flashinfer():
    import torch, flashinfer
    q = torch.randn(4, 8, 128, device="cuda", dtype=torch.float16)
    k = torch.randn(256, 8, 128, device="cuda", dtype=torch.float16)
    flashinfer.single_prefill_with_kv_cache(q, k, k.clone(), causal=False); torch.cuda.synchronize()
def t_vllm():
    import torch, vllm, vllm._C_stable_libtorch  # noqa
getattr(sys.modules[__name__], "t_" + which)()
pat = re.compile(r"lib(cudart|nvrtc|nvrtc-builtins|nvJitLink|nvvm|cublas|cublasLt|cudnn|nccl|nvshmem_host|cuda)\.so")
libs = sorted({l.split()[-1] for l in open("/proc/self/maps") if len(l.split()) >= 6 and pat.search(l.split()[-1])})
for p in libs:
    tag = "PIP-WHEEL" if "site-packages/nvidia" in p else ("SOFT" if p.startswith("/soft/") else "OTHER")
    print(f"[{which}] {tag:9s} {p}")
