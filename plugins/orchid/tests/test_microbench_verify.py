import argparse
import os
import statistics
import time
from contextlib import nullcontext

import torch

from orchid.llmscheduler.runtime.base import AttentionContext
from orchid.llmscheduler.runtime.trt_runtime import TensorRTModelRuntime


def _bench_trt(
    model_path: str,
    tokenizer_path: str,
    *,
    engine_path: str | None,
    fp16: bool,
    bs: int,
    prefill_len: int,
    decode_steps: int,
    warmup_steps: int,
    device: str,
):
    from transformers import AutoTokenizer

    from orchid.llmscheduler.model_params import infer_model_params

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    runtime = TensorRTModelRuntime(model_path, use_fp16=fp16, engine_path=engine_path)
    mp = infer_model_params(model_path, tokenizer_path)
    page_size = int(mp.page_size)
    num_layers = int(mp.num_layers)
    pages_per_req = int((int(prefill_len) + int(decode_steps) + int(page_size) - 1) // int(page_size))
    pages_per_layer_need = int(int(bs) * int(max(1, pages_per_req)))
    max_pages = int(max(int(mp.max_pages), int(pages_per_layer_need) * int(num_layers)))
    # Force larger max_pages to avoid OOM in microbenchmark
    max_pages = max(max_pages, 20000)
    ctx = AttentionContext(
        num_layers=int(mp.num_layers),
        num_heads=int(mp.num_heads),
        kv_num_heads=int(mp.kv_num_heads),
        head_dim=int(mp.head_dim),
        page_size=page_size,
        max_pages=max_pages,
        use_cpp_metadata=True,
        device=device,
        use_fp16=fp16,
    )

    max_total_tokens = 4096
    spec = os.environ.get("LLMSCHEDULER_TRT_INPUT_IDS_PROFILES", "").strip()
    if spec:
        try:
            mx = []
            for part in spec.split(";"):
                cols = [c.strip() for c in part.split(",")]
                if len(cols) == 3:
                    mx.append(int(cols[2]))
            if mx:
                max_total_tokens = int(max(mx))
        except Exception:
            max_total_tokens = 4096
    max_total_tokens = int(max(1, max_total_tokens))
    max_prefill_len = int(max(1, max_total_tokens // int(max(1, int(bs)))))
    prefill_len = int(min(int(prefill_len), int(max_prefill_len)))

    vocab = int(getattr(tokenizer, "vocab_size", 32000)) or 32000
    prompt = torch.randint(low=0, high=vocab, size=(int(bs), int(prefill_len)), dtype=torch.int64, device="cpu")
    input_ids = prompt.reshape(-1).contiguous()

    ctx.current_batch_req_ids = list(range(int(bs)))
    ctx.current_batch_seq_lens = [int(prefill_len)] * int(bs)
    ctx.current_batch_total_lens = [int(prefill_len)] * int(bs)
    ctx.current_batch_history_lens = [0] * int(bs) # Added history lens
    ctx.current_batch_is_prefill = [True] * int(bs)
    ctx.is_all_decode = False

    use_non_default_stream = bool(int(os.environ.get("LLMSCHEDULER_TRT_NON_DEFAULT_STREAM", "0")))
    stream = torch.cuda.Stream() if use_non_default_stream and str(device).startswith("cuda") else None
    ctxmgr = torch.cuda.stream(stream) if stream is not None else nullcontext()

    with ctxmgr:
        _ = runtime.forward(input_ids.to(device, dtype=torch.int32, non_blocking=True), ctx)

    for i in range(int(warmup_steps)):
        tok = torch.randint(low=0, high=vocab, size=(int(bs),), dtype=torch.int32, device=device)
        ctx.current_batch_seq_lens = [1] * int(bs)
        ctx.current_batch_total_lens = [int(prefill_len) + i + 1] * int(bs)
        ctx.current_batch_history_lens = [int(prefill_len) + i] * int(bs) # Added history lens
        ctx.current_batch_is_prefill = [False] * int(bs)
        ctx.is_all_decode = True
        with ctxmgr:
            _ = runtime.forward(tok, ctx)

    times_ms = []
    for i in range(int(decode_steps)):
        tok = torch.randint(low=0, high=vocab, size=(int(bs),), dtype=torch.int32, device=device)
        ctx.current_batch_seq_lens = [1] * int(bs)
        ctx.current_batch_total_lens = [int(prefill_len) + int(warmup_steps) + i + 1] * int(bs)
        ctx.current_batch_history_lens = [int(prefill_len) + int(warmup_steps) + i] * int(bs) # Added history lens
        ctx.current_batch_is_prefill = [False] * int(bs)
        ctx.is_all_decode = True
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with ctxmgr:
            logits = runtime.forward(tok, ctx)
        torch.cuda.synchronize()
        times_ms.append((time.perf_counter() - t0) * 1000.0)
        
        # Correctness check: ensure logits are not all zero or NaN
        if i == 0:
            if torch.isnan(logits).any():
                print("[VERIFY] Error: TRT logits contain NaN")
            if torch.all(logits == 0):
                print("[VERIFY] Warning: TRT logits are all zero")
            print(f"[VERIFY] Logits sample (first 5): {logits[0, :5].tolist()}")

    mean_ms = float(statistics.mean(times_ms)) if times_ms else 0.0
    tok_s = (float(bs) * float(decode_steps)) / (float(sum(times_ms)) / 1000.0) if times_ms else 0.0
    
    # Cleanup to free GPU memory for vLLM
    del ctx
    del runtime
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    
    return {"decode_step_ms_mean": mean_ms, "decode_tok_s": tok_s}


def _bench_vllm(tokenizer_path: str, *, bs: int, prefill_len: int, decode_steps: int, fp16: bool):
    try:
        from vllm import LLM, SamplingParams

        model = tokenizer_path
        dtype = "float16" if fp16 else "float32"
        llm = LLM(
            model=model,
            tokenizer=tokenizer_path,
            dtype=dtype,
            trust_remote_code=True,
            max_model_len=max(4096, int(prefill_len) + int(decode_steps) + 16),
            gpu_memory_utilization=0.4,
            disable_log_stats=True,
        )
        sampling = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=int(decode_steps))

        prompts = ["x"] * int(bs)
        t0 = time.perf_counter()
        _ = llm.generate(prompts, sampling)
        dt = time.perf_counter() - t0
        tok_s = (float(bs) * float(decode_steps)) / float(dt) if dt > 0 else 0.0
        return {"decode_tok_s": tok_s}
    except Exception as e:
        return {"error": str(e)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--engine", default="")
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--prefill-len", type=int, default=512)
    ap.add_argument("--decode-steps", type=int, default=128)
    ap.add_argument("--warmup-steps", type=int, default=10)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    engine_path = str(args.engine).strip() or None
    trt = _bench_trt(
        args.model,
        args.tokenizer,
        engine_path=engine_path,
        fp16=bool(args.fp16),
        bs=int(args.bs),
        prefill_len=int(args.prefill_len),
        decode_steps=int(args.decode_steps),
        warmup_steps=int(args.warmup_steps),
        device=str(args.device),
    )
    vllm = _bench_vllm(args.tokenizer, bs=int(args.bs), prefill_len=int(args.prefill_len), decode_steps=int(args.decode_steps), fp16=bool(args.fp16))

    print({"trt": trt, "vllm": vllm}, flush=True)


if __name__ == "__main__":
    main()
