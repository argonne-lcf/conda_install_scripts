import os, sys, socket, traceback
import h5py  # deliberately before mpi4py: regression check for GTL-less h5py abort
from mpi4py import MPI
comm = MPI.COMM_WORLD; rank, size = comm.rank, comm.size
host = socket.gethostname()
def log(*a):
    print(f"[rank {rank} {host}]", *a, flush=True)
def section(name, fn):
    try:
        r = fn(); log(name, "OK", r if r is not None else "")
    except Exception as e:
        log(name, "FAIL", repr(e)); 
        if rank == 0: traceback.print_exc()

log("MPI lib:", MPI.Get_library_version().strip().splitlines()[0][:70], "size", size)
local_rank = comm.Split_type(MPI.COMM_TYPE_SHARED).rank
log("local_rank", local_rank, "PALS_LOCAL_RANKID=", os.environ.get("PALS_LOCAL_RANKID"))
log("CVD=", os.environ.get("CUDA_VISIBLE_DEVICES"))

def t_torch():
    import torch
    assert torch.cuda.is_available()
    d = torch.device("cuda")
    a = torch.randn(1024,1024, device=d); b = (a@a).sum().item()
    return f"torch {torch.__version__} cuda {torch.version.cuda} dev {torch.cuda.get_device_name(0)} sum={b:.1f}"
section("torch.cuda", t_torch)

def t_cupy_mpi():
    import cupy as cp
    x = cp.ones(1000, dtype=cp.float32); y = cp.empty_like(x)
    comm.Allreduce(x, y, op=MPI.SUM)   # CUDA-aware MPI through GTL
    assert float(y[0]) == size
    return f"cupy {cp.__version__} GPU allreduce sum={float(y[0])}"
section("mpi4py CUDA-aware allreduce (cupy)", t_cupy_mpi)

def t_torch_dist():
    import torch, torch.distributed as dist
    master = comm.bcast(host if rank == 0 else None, root=0)
    os.environ.update(MASTER_ADDR=master, MASTER_PORT="29517", RANK=str(rank), WORLD_SIZE=str(size))
    dist.init_process_group("nccl", rank=rank, world_size=size, device_id=torch.device("cuda:0"))
    t = torch.ones(1<<20, device="cuda"); dist.all_reduce(t)
    assert t[0].item() == size
    ver = torch.cuda.nccl.version()
    dist.destroy_process_group()
    return f"nccl {ver} allreduce over {size} ranks ok"
section("torch.distributed NCCL", t_torch_dist)

def t_torch_mpi():
    import torch, torch.distributed as dist
    dist.init_process_group("mpi")
    t = torch.ones(10, device="cuda"); dist.all_reduce(t)
    assert t[0].item() == size
    dist.destroy_process_group()
    return "torch mpi backend GPU allreduce ok"
section("torch.distributed MPI backend", t_torch_mpi)

if rank == 0:
    def t_tf():
        import tensorflow as tf
        g = tf.config.list_physical_devices("GPU")
        with tf.device("/GPU:0"):
            r = tf.reduce_sum(tf.matmul(tf.ones((512,512)), tf.ones((512,512)))).numpy()
        return f"tf {tf.__version__} gpus={len(g)} r={r}"
    section("tensorflow GPU", t_tf)
    def t_jax():
        import jax, jax.numpy as jnp
        d = jax.devices()
        return f"jax {jax.__version__} devices={d} r={float(jnp.ones((256,256)).sum())}"
    section("jax GPU", t_jax)
    def t_fa():
        import torch, flash_attn
        from flash_attn import flash_attn_func
        q = torch.randn(1,128,8,64, device="cuda", dtype=torch.float16)
        o = flash_attn_func(q,q,q); torch.cuda.synchronize()
        return f"flash_attn {flash_attn.__version__} out {tuple(o.shape)}"
    section("flash_attn", t_fa)
    def t_te():
        import torch, transformer_engine.pytorch as te
        l = te.Linear(64,64).cuda(); y = l(torch.randn(8,64,device="cuda")); torch.cuda.synchronize()
        return f"TE linear {tuple(y.shape)}"
    section("transformer_engine", t_te)
    def t_ds():
        import deepspeed; from deepspeed.ops.adam import FusedAdam
        import torch; p = torch.nn.Parameter(torch.randn(10, device="cuda")); FusedAdam([p]).step()
        return f"deepspeed {deepspeed.__version__} FusedAdam ok"
    section("deepspeed", t_ds)
    def t_vllm():
        import vllm; return vllm.__version__
    section("vllm import", t_vllm)
    def t_compile():
        import torch
        f = torch.compile(lambda x: (x*2).sin())
        return f"torch.compile/triton {f(torch.randn(64,device='cuda')).shape}"
    section("torch.compile (triton JIT)", t_compile)
    def t_mpi4jax():
        import jax, jax.numpy as jnp, mpi4jax
        return "mpi4jax import ok"
    section("mpi4jax", t_mpi4jax)
def t_h5py():
    fn = f"./h5test_{os.environ.get('PBS_JOBID','x').split('.')[0]}.h5"
    with h5py.File(fn, "w", driver="mpio", comm=comm) as f:
        d = f.create_dataset("x", (size, 4), dtype="i4"); d[rank] = rank
    with h5py.File(fn, "r", driver="mpio", comm=comm) as f:
        assert list(f["x"][:,0]) == list(range(size))
    comm.Barrier()
    if rank == 0: os.remove(fn)
    return f"h5py {h5py.__version__} hdf5 {h5py.version.hdf5_version} parallel write/read ok"
section("h5py parallel MPI-IO", t_h5py)
comm.Barrier()
log("done")
