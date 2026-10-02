# 2-node performance benchmarks (2x4 A100-40GB, bf16)

Synthetic-data throughput for DDP (ResNet-50), FSDP2/HSDP (1.42B GPT), and vLLM offline
inference (Qwen2.5-7B-Instruct, 512 GSM8K prompts, 256 tokens, `ignore_eos`), single node
and across both nodes (Ray, TP=4 PP=2).

```
cd alcf_polaris/tests/perf     # must be on a shared filesystem
qsub perf-2026-09-17.pbs       # debug queue, ~20 min; per-run logs in logs/<jobid>/
```

Needs the model in `~/.cache/huggingface` (jobs run with `HF_HUB_OFFLINE=1`).

## Results

conda/2026-09-17 (torch 2.14.0, NCCL 2.30.7, vLLM 0.29.1, Ray 2.58.0), job 7646687, full
logs in `logs/7646687/`. The module has no NCCL net plugin, so inter-node traffic uses
TCP sockets. 2025-09-25 default-module numbers from earlier runs are shown for reference.

| test | 2026-09-17 | 2025-09-25 |
|---|---|---|
| DDP ResNet-50, 1 GPU | 1880 img/s | 1860 |
| DDP ResNet-50, 4 GPUs | 7159 img/s | 7092 |
| DDP ResNet-50, 8 GPUs | 5633 img/s | 5671-5936 |
| FSDP2 1.42B, 4 GPUs | 73.1k tok/s, 166 TFLOPS/GPU | 72k, 165 |
| FSDP2 1.42B, 8 GPUs full shard | 5.1k tok/s | 5.1-5.5k |
| FSDP2 1.42B, 8 GPUs HSDP (2x4) | 11.6k tok/s | - |
| vLLM TP=4, 1 node (mp) | 22.9k gen tok/s, init 104 s | 14.4k |
| vLLM TP=4 PP=2, 2 nodes (ray) | 5.0k gen tok/s, init 251 s | 0.66k, init 193 s |

Over TCP, 2-node training is slower in total than 1 node except HSDP; prefer single-node
TP for models that fit.

## Gotchas

- `TMPDIR=/tmp` and `ray start --temp-dir=/tmp/ray_$USER`: PBS's per-job TMPDIR makes Ray's
  AF_UNIX socket paths exceed the 107-byte limit.
- `RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1`: otherwise vLLM Ray workers fail
  `init_device` with "invalid device ordinal" (ranks already pinned by the launcher env).
- `perf_vllm.py` must keep its `__main__` guard; vLLM's mp workers re-import the script.
- TP=8 is invalid for Qwen2.5-7B (28 attention heads).
