import argparse
import os
from dataclasses import dataclass

import torch

from orchid.llmscheduler.model_params import infer_model_params
from orchid.llmscheduler.runtime.base import AttentionContext
from orchid.llmscheduler.runtime.trt_runtime import TensorRTModelRuntime


@dataclass(frozen=True)
class Scenario:
    name: str
    bs: int
    prefill_len: int
    decode_steps: int


def _scenarios() -> list[Scenario]:
    return [
        Scenario("bs1_p16_s8", 1, 16, 8),
        Scenario("bs4_p128_s32", 4, 128, 32),
        Scenario("bs10_p128_s128", 10, 128, 128),
    ]


def _tokenizer_vocab(tokenizer_path: str) -> int:
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        v = int(getattr(tok, "vocab_size", 0) or 0)
        return v if v > 0 else 151936
    except Exception:
        return 151936


def _make_ctx(mp, *, bs: int, prefill_len: int, decode_steps: int, device: str, fp16: bool) -> AttentionContext:
    page_size = int(mp.page_size)
    pages_per_req = int((int(prefill_len) + int(decode_steps) + int(page_size) - 1) // int(page_size))
    pages_per_layer_need = int(int(bs) * int(max(1, pages_per_req)))
    max_pages = int(max(int(mp.max_pages), int(pages_per_layer_need) * int(mp.num_layers)))
    max_pages = max(max_pages, 20000)
    return AttentionContext(
        num_layers=int(mp.num_layers),
        num_heads=int(mp.num_heads),
        kv_num_heads=int(mp.kv_num_heads),
        head_dim=int(mp.head_dim),
        page_size=int(page_size),
        max_pages=int(max_pages),
        use_cpp_metadata=True,
        device=str(device),
        use_fp16=bool(fp16),
    )


def _trt_prefill_logits(runtime: TensorRTModelRuntime, ctx: AttentionContext, prompt_cpu_i64: torch.Tensor) -> torch.Tensor:
    bs, prefill_len = int(prompt_cpu_i64.shape[0]), int(prompt_cpu_i64.shape[1])
    input_ids = prompt_cpu_i64.reshape(-1).contiguous().to(ctx.device, dtype=torch.int32)
    ctx.current_batch_req_ids = list(range(int(bs)))
    ctx.current_batch_seq_lens = [int(prefill_len)] * int(bs)
    ctx.current_batch_total_lens = [int(prefill_len)] * int(bs)
    ctx.current_batch_history_lens = [0] * int(bs)
    ctx.current_batch_is_prefill = [True] * int(bs)
    ctx.is_all_decode = False
    _ = runtime.forward(input_ids, ctx)
    return runtime.forward(input_ids, ctx)


def _trt_extract_last_logits(prefill_logits: torch.Tensor, *, bs: int, prefill_len: int) -> torch.Tensor:
    if prefill_logits.dim() != 2:
        raise RuntimeError(f"Unexpected TRT logits dim: {tuple(prefill_logits.shape)}")
    if int(prefill_logits.shape[0]) == int(bs) * int(prefill_len):
        last = []
        for i in range(int(bs)):
            idx = (i + 1) * int(prefill_len) - 1
            last.append(prefill_logits[idx])
        return torch.stack(last, dim=0)
    if int(prefill_logits.shape[0]) == int(bs):
        return prefill_logits
    raise RuntimeError(f"Unexpected TRT logits shape: {tuple(prefill_logits.shape)} (bs={bs}, prefill_len={prefill_len})")


def _trt_greedy_generate(runtime: TensorRTModelRuntime, ctx: AttentionContext, prompt_cpu_i64: torch.Tensor, *, decode_steps: int) -> list[list[int]]:
    bs, prefill_len = int(prompt_cpu_i64.shape[0]), int(prompt_cpu_i64.shape[1])
    prefill_logits = _trt_prefill_logits(runtime, ctx, prompt_cpu_i64)
    last_logits = _trt_extract_last_logits(prefill_logits, bs=bs, prefill_len=prefill_len)
    tok = torch.argmax(last_logits, dim=-1).to(ctx.device, dtype=torch.int32)
    out: list[list[int]] = [[int(x)] for x in tok.detach().cpu().tolist()]

    history = [int(prefill_len)] * int(bs)
    total = [int(prefill_len) + 1] * int(bs)
    ctx.current_batch_req_ids = list(range(int(bs)))
    for step in range(1, int(decode_steps)):
        ctx.current_batch_seq_lens = [1] * int(bs)
        ctx.current_batch_history_lens = list(history)
        ctx.current_batch_total_lens = list(total)
        ctx.current_batch_is_prefill = [False] * int(bs)
        ctx.is_all_decode = True
        logits = runtime.forward(tok, ctx)
        if logits.dim() != 2 or int(logits.shape[0]) != int(bs):
            raise RuntimeError(f"Unexpected TRT decode logits shape: {tuple(logits.shape)}")
        tok = torch.argmax(logits, dim=-1).to(ctx.device, dtype=torch.int32)
        ids = tok.detach().cpu().tolist()
        for i in range(int(bs)):
            out[i].append(int(ids[i]))
        history = [int(h) + 1 for h in history]
        total = [int(t) + 1 for t in total]
    return out


def _vllm_greedy_generate(llm, prompts_token_ids: list[list[int]], *, decode_steps: int) -> list[list[int]]:
    from vllm import SamplingParams

    sampling = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=int(decode_steps), ignore_eos=True)
    outputs = llm.generate([{"prompt_token_ids": ids} for ids in prompts_token_ids], sampling)
    out: list[list[int]] = []
    for o in outputs:
        out.append([int(x) for x in o.outputs[0].token_ids])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--engine", default="")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--eager", action="store_true")
    ap.add_argument("--vllm-gpu-mem", type=float, default=0.25)
    ap.add_argument("--min-match-rate", type=float, default=0.95)
    args = ap.parse_args()

    torch.manual_seed(int(args.seed))
    engine_path = str(args.engine).strip() or None
    vocab = _tokenizer_vocab(args.tokenizer)

    mp = infer_model_params(args.model, args.tokenizer)
    runtime = TensorRTModelRuntime(args.model, use_fp16=bool(args.fp16), engine_path=engine_path)

    from vllm import LLM

    dtype = "float16" if bool(args.fp16) else "float32"
    max_prefill = max(int(s.prefill_len) for s in _scenarios())
    max_steps = max(int(s.decode_steps) for s in _scenarios())
    llm = LLM(
        model=args.tokenizer,
        tokenizer=args.tokenizer,
        dtype=dtype,
        trust_remote_code=True,
        max_model_len=max(4096, int(max_prefill) + int(max_steps) + 32),
        gpu_memory_utilization=float(args.vllm_gpu_mem),
        disable_log_stats=True,
        enforce_eager=bool(args.eager),
    )

    failures = 0
    for sc in _scenarios():
        ctx = _make_ctx(mp, bs=int(sc.bs), prefill_len=int(sc.prefill_len), decode_steps=int(sc.decode_steps), device=str(args.device), fp16=bool(args.fp16))
        prompt = torch.randint(low=0, high=int(vocab), size=(int(sc.bs), int(sc.prefill_len)), dtype=torch.int64, device="cpu")
        trt = _trt_greedy_generate(runtime, ctx, prompt, decode_steps=int(sc.decode_steps))
        vllm = _vllm_greedy_generate(llm, prompt.tolist(), decode_steps=int(sc.decode_steps))
        mismatch = 0
        for i in range(int(sc.bs)):
            for t in range(int(sc.decode_steps)):
                if int(trt[i][t]) != int(vllm[i][t]):
                    mismatch += 1
        total = int(sc.bs) * int(sc.decode_steps)
        rate = float(total - mismatch) / float(max(1, total))
        ok = float(rate) >= float(args.min_match_rate)
        print({"scenario": sc.name, "match": bool(ok), "token_match_rate": float(rate), "mismatch": int(mismatch)}, flush=True)
        if not ok:
            failures += 1

    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
