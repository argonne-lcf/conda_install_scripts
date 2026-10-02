"""vLLM offline throughput. --backend mp (single node) or ray (multinode, cluster must be up)."""
import argparse, os, time, sys
p = argparse.ArgumentParser(); p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
p.add_argument("--tp", type=int, default=4); p.add_argument("--pp", type=int, default=1); p.add_argument("--backend", default="mp")
p.add_argument("--enforce-eager", action="store_true"); p.add_argument("--n", type=int, default=512); p.add_argument("--max-tokens", type=int, default=256); a = p.parse_args()
def main():
    from vllm import LLM, SamplingParams
    try:
        from datasets import load_dataset
        qs = [r["question"] for r in load_dataset("openai/gsm8k", "main", split="test")][: a.n]
    except Exception as e:
        print("gsm8k load failed, using synthetic prompts:", repr(e)); qs = [f"Explain topic number {i} in detail." for i in range(a.n)]
    t0 = time.time()
    llm = LLM(model=a.model, tensor_parallel_size=a.tp, pipeline_parallel_size=a.pp, distributed_executor_backend=a.backend,
              dtype="bfloat16", max_model_len=2048, gpu_memory_utilization=0.85, trust_remote_code=False, enforce_eager=a.enforce_eager)
    print(f"engine init {time.time()-t0:.0f}s", flush=True)
    tok = llm.get_tokenizer()
    prompts = [tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True) for q in qs]
    sp = SamplingParams(temperature=0.0, max_tokens=a.max_tokens, ignore_eos=True)
    llm.generate(prompts[:8], sp)  # warmup
    t0 = time.time(); out = llm.generate(prompts, sp); dt = time.time() - t0
    gen = sum(len(o.outputs[0].token_ids) for o in out); inp = sum(len(o.prompt_token_ids) for o in out)
    print(f"RESULT vllm {a.model} backend={a.backend} tp={a.tp} pp={a.pp} reqs={len(out)} time_s={dt:.1f} "
          f"gen_tok/s={gen/dt:.0f} total_tok/s={(gen+inp)/dt:.0f} req/s={len(out)/dt:.2f}", flush=True)
    print("sample:", out[0].outputs[0].text[:200].replace("\n", " "))

if __name__ == "__main__":  # vLLM spawns the engine core; unguarded module code re-runs in the child
    main()
