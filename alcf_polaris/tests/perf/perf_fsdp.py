"""GPT-style model with FSDP2 (fully_shard), synthetic tokens. Run with mpiexec -n {4,8} --ppn 4 ./gpu_wrap.sh python perf_fsdp.py"""
import argparse, math, statistics, torch, torch.nn as nn, torch.nn.functional as F, torch.distributed as dist
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from common import init_dist, log, Timer
p = argparse.ArgumentParser()
p.add_argument("--d", type=int, default=2048); p.add_argument("--layers", type=int, default=24); p.add_argument("--heads", type=int, default=16)
p.add_argument("--vocab", type=int, default=50304); p.add_argument("--seq", type=int, default=1024); p.add_argument("--bs", type=int, default=4)
p.add_argument("--hsdp", action="store_true", help="shard within node (4 GPUs), replicate across nodes"); p.add_argument("--steps", type=int, default=20); p.add_argument("--warmup", type=int, default=5); a = p.parse_args()
comm, rank, size = init_dist()
assert size > 1, "FSDP test needs >1 rank"

class Block(nn.Module):
    def __init__(s, d, h):
        super().__init__(); s.h = h; s.ln1 = nn.LayerNorm(d); s.ln2 = nn.LayerNorm(d)
        s.qkv = nn.Linear(d, 3*d, bias=False); s.proj = nn.Linear(d, d, bias=False)
        s.fc1 = nn.Linear(d, 4*d, bias=False); s.fc2 = nn.Linear(4*d, d, bias=False)
    def forward(s, x):
        B, T, C = x.shape
        q, k, v = s.qkv(s.ln1(x)).view(B, T, 3, s.h, C // s.h).permute(2, 0, 3, 1, 4)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(B, T, C)
        x = x + s.proj(y)
        return x + s.fc2(F.gelu(s.fc1(s.ln2(x))))
class GPT(nn.Module):
    def __init__(s, a):
        super().__init__(); s.wte = nn.Embedding(a.vocab, a.d); s.wpe = nn.Embedding(a.seq, a.d)
        s.blocks = nn.ModuleList([Block(a.d, a.heads) for _ in range(a.layers)]); s.ln = nn.LayerNorm(a.d)
        s.head = nn.Linear(a.d, a.vocab, bias=False)
    def forward(s, idx):
        x = s.wte(idx) + s.wpe(torch.arange(idx.shape[1], device=idx.device))
        for b in s.blocks: x = b(x)
        return s.head(s.ln(x))

torch.manual_seed(0)
with torch.device("cuda"): model = GPT(a)
nparams = sum(p.numel() for p in model.parameters())
mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
mesh = None
if a.hsdp:
    from torch.distributed.device_mesh import init_device_mesh
    mesh = init_device_mesh("cuda", (size // 4, 4), mesh_dim_names=("replicate", "shard"))
for b in model.blocks: fully_shard(b, mp_policy=mp, mesh=mesh)
fully_shard(model, mp_policy=mp, mesh=mesh)
opt = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=True)
x = torch.randint(0, a.vocab, (a.bs, a.seq), device="cuda"); tm = Timer()
for i in range(a.warmup + a.steps):
    with (tm if i >= a.warmup else Timer()):
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, a.vocab).float(), x.view(-1))
        loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
dist.barrier()
step = statistics.median(tm.t); tok = a.bs * a.seq / step
flops = 6 * nparams * a.bs * a.seq + 12 * a.layers * a.d * a.seq**2 * a.bs  # fwd+bwd, incl attention
tflops = flops / step / 1e12; mem = torch.cuda.max_memory_allocated() / 2**30
log(rank, f"RESULT {'hsdp' if a.hsdp else 'fsdp2'} gpt params={nparams/1e9:.2f}B gpus={size} bs/gpu={a.bs} seq={a.seq} step_ms={step*1e3:.0f} "
          f"tok/s/gpu={tok:.0f} tok/s_total={tok*size:.0f} TFLOPS/gpu={tflops:.0f} peak_mem_GB={mem:.1f} loss={loss.item():.2f}")
dist.destroy_process_group()
