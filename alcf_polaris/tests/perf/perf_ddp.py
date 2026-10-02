"""ResNet-50 DDP synthetic training throughput. Run with mpiexec -n {1,4,8} --ppn 4 ./gpu_wrap.sh python perf_ddp.py"""
import argparse, torch, torch.nn as nn, torch.distributed as dist, torchvision, statistics
from torch.nn.parallel import DistributedDataParallel as DDP
from common import init_dist, log, Timer
p = argparse.ArgumentParser(); p.add_argument("--bs", type=int, default=128); p.add_argument("--steps", type=int, default=40)
p.add_argument("--warmup", type=int, default=10); a = p.parse_args()
comm, rank, size = init_dist()
torch.backends.cudnn.benchmark = True
model = torchvision.models.resnet50().cuda().to(memory_format=torch.channels_last)
if size > 1: model = DDP(model, device_ids=[0], bucket_cap_mb=25)
opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
x = torch.randn(a.bs, 3, 224, 224, device="cuda").to(memory_format=torch.channels_last)
y = torch.randint(0, 1000, (a.bs,), device="cuda")
loss_fn = nn.CrossEntropyLoss(); tm = Timer()
for i in range(a.warmup + a.steps):
    with (tm if i >= a.warmup else Timer()):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = loss_fn(model(x), y)
        loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
if size > 1: dist.barrier()
step = statistics.median(tm.t); per_gpu = a.bs / step; agg = per_gpu * size
mem = torch.cuda.max_memory_allocated() / 2**30
log(rank, f"RESULT ddp resnet50 bf16 gpus={size} bs/gpu={a.bs} step_ms={step*1e3:.1f} img/s/gpu={per_gpu:.0f} img/s_total={agg:.0f} peak_mem_GB={mem:.1f}")
if size > 1: dist.destroy_process_group()
