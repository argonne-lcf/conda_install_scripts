import os, socket, time, torch, torch.distributed as dist
from mpi4py import MPI

def init_dist():
    comm = MPI.COMM_WORLD
    rank, size = comm.rank, comm.size
    master = comm.bcast(socket.gethostname() if rank == 0 else None, root=0)
    os.environ.update(MASTER_ADDR=master, MASTER_PORT=os.environ.get("MASTER_PORT", "29517"),
                      RANK=str(rank), WORLD_SIZE=str(size))
    torch.cuda.set_device(0)  # CUDA_VISIBLE_DEVICES set by gpu_wrap.sh
    if size > 1:
        dist.init_process_group("nccl", rank=rank, world_size=size, device_id=torch.device("cuda:0"))
    return comm, rank, size

def log(rank, *a):
    if rank == 0: print(*a, flush=True)

class Timer:
    def __init__(self): self.t = []
    def __enter__(self): torch.cuda.synchronize(); self.t0 = time.perf_counter(); return self
    def __exit__(self, *e): torch.cuda.synchronize(); self.t.append(time.perf_counter() - self.t0)
