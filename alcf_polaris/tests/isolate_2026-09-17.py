import sys, faulthandler, traceback
faulthandler.enable()
which = sys.argv[1]
def t_fa():
    import torch, flash_attn
    from flash_attn import flash_attn_func
    q = torch.randn(1,128,8,64, device="cuda", dtype=torch.float16)
    o = flash_attn_func(q,q,q); torch.cuda.synchronize()
    return f"flash_attn {flash_attn.__version__} out {tuple(o.shape)}"
def t_fa_after_tf():
    import tensorflow as tf, jax, jax.numpy as jnp
    tf.reduce_sum(tf.ones((8,8))).numpy(); float(jnp.ones(8).sum())
    return t_fa()
def t_te():
    import torch, transformer_engine.pytorch as te
    l = te.Linear(64,64).cuda(); y = l(torch.randn(8,64,device="cuda")); torch.cuda.synchronize()
    return f"TE linear {tuple(y.shape)}"
def t_ds():
    import deepspeed; from deepspeed.ops.adam import FusedAdam
    import torch; p = torch.nn.Parameter(torch.randn(10, device="cuda")); FusedAdam([p]).step()
    return f"deepspeed {deepspeed.__version__} FusedAdam ok"
def t_vllm():
    import vllm; return f"vllm {vllm.__version__}"
def t_compile():
    import torch
    f = torch.compile(lambda x: (x*2).sin())
    return f"torch.compile/triton {f(torch.randn(64,device='cuda')).shape}"
def t_mpi4jax():
    import jax, jax.numpy as jnp, mpi4jax
    return "mpi4jax import ok"
def t_ort():
    import onnxruntime as ort; return f"onnxruntime {ort.__version__} {ort.get_available_providers()}"
def t_xgb():
    import xgboost as xgb, numpy as np
    d = xgb.DMatrix(np.random.rand(100,4), label=np.random.rand(100))
    xgb.train({"device":"cuda","tree_method":"hist"}, d, 2); return f"xgboost {xgb.__version__} gpu hist ok"
def t_mamba():
    import torch; from mamba_ssm import Mamba
    m = Mamba(d_model=64, d_state=16, d_conv=4, expand=2).cuda(); y = m(torch.randn(1,16,64,device="cuda")); torch.cuda.synchronize()
    return f"mamba_ssm out {tuple(y.shape)}"
def t_flashinfer():
    import flashinfer; return f"flashinfer {flashinfer.__version__}"
def t_gce():
    import globus_compute_endpoint, parsl; return f"gce {globus_compute_endpoint.__version__} parsl {parsl.__version__}"
def t_apex():
    import torch, apex, amp_C, fused_layer_norm_cuda
    from apex.normalization import FusedLayerNorm
    from apex.optimizers import FusedAdam
    ln = FusedLayerNorm(64).cuda(); y = ln(torch.randn(8,64,device="cuda")); y.sum().backward()
    FusedAdam(ln.parameters()).step(); torch.cuda.synchronize()
    return f"apex FusedLayerNorm fwd/bwd + FusedAdam step ok, out {tuple(y.shape)}"
def t_numpyro():
    import jax, numpyro, numpyro.distributions as dist
    from numpyro.infer import MCMC, NUTS
    def model(): numpyro.sample("x", dist.Normal(0., 1.))
    m = MCMC(NUTS(model), num_warmup=50, num_samples=50, progress_bar=False); m.run(jax.random.PRNGKey(0))
    return f"numpyro {numpyro.__version__} NUTS on {jax.devices()[0].platform}, mean {float(m.get_samples()['x'].mean()):.2f}"
def t_mcore():
    import megatron.core, megatron.core.models.gpt.gpt_layer_specs  # noqa (FA4 flash_attn.cute import)
    return f"megatron-core {megatron.core.__version__} gpt_layer_specs ok"
def t_verl():
    import verl, verl.trainer.main_ppo, verl.trainer.ppo.ray_trainer, verl.workers.engine_workers  # noqa (orjson, megatron)
    return f"verl {verl.__version__} trainer/engine imports ok"
def t_rl():
    import trl, ray.rllib
    return f"trl {trl.__version__} rllib ok"
def t_titan():
    # torchtitan caps datasets<4.8 and is installed --no-deps next to datasets 5.x: exercise its HF
    # text dataloader (load_dataset, split_dataset_by_node, streaming state_dict resume) on a local
    # dataset shaped like its c4_test asset. The tokenizer is a byte-level stub.
    import json, os, shutil, tempfile, datasets, torchtitan, torchtitan.train  # noqa
    from datasets import load_dataset
    from torchtitan.components.tokenizer import BaseTokenizer
    from torchtitan.hf_datasets import text_datasets as td
    class Bytes(BaseTokenizer):
        def encode(self, s, add_bos=True, add_eos=True): return [1]*add_bos + list(s.encode()) + [2]*add_eos
        def decode(self, ids): return bytes(i for i in ids if i > 2).decode()
        def get_vocab_size(self): return 259
    d = tempfile.mkdtemp(prefix="titan-", dir=os.getcwd())
    with open(f"{d}/train.json", "w") as f:
        for i in range(64): f.write(json.dumps({"text": f"document {i} " + "lorem ipsum " * (i % 7 + 1)}) + "\n")
    cfg = td.DATASETS["c4_test"]
    out = []
    for name, loader in [("map", lambda p: load_dataset(p, split="train")),
                         ("stream", lambda p: load_dataset(p, split="train", streaming=True))]:
        td.DATASETS["c4_test"] = type(cfg)(**{**cfg.__dict__, "path": d, "loader": loader})
        mk = lambda: td.HuggingFaceTextDataset("c4_test", None, Bytes(), seq_len=32, dp_rank=1, dp_world_size=2)
        a = mk(); it = iter(a); [next(it) for _ in range(5)]; sd = a.state_dict(); want = next(it)[1]
        b = mk(); b.load_state_dict(sd); got = next(iter(b))[1]
        assert (want == got).all(), f"{name}: resume mismatch"
        out.append(name)
    td.DATASETS["c4_test"] = cfg; shutil.rmtree(d)
    return f"torchtitan {torchtitan.__version__} datasets {datasets.__version__} HF dataloader {'+'.join(out)} resume ok"
tests = dict(fa=t_fa, fa_after_tf=t_fa_after_tf, te=t_te, ds=t_ds, vllm=t_vllm, compile=t_compile, mpi4jax=t_mpi4jax, ort=t_ort, xgb=t_xgb, mamba=t_mamba, flashinfer=t_flashinfer, gce=t_gce, apex=t_apex, numpyro=t_numpyro, mcore=t_mcore, verl=t_verl, rl=t_rl, titan=t_titan)
try:
    print(f"[{which}] OK", tests[which](), flush=True)
except Exception as e:
    print(f"[{which}] FAIL {e!r}", flush=True); traceback.print_exc()
