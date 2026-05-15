import argparse
import os
import statistics
import time
from collections import defaultdict

os.environ["LLMSCHEDULER_PROFILE"] = "1"
os.environ["LLMSCHEDULER_PROFILE_DETAIL"] = "1"
os.environ["LLMSCHEDULER_PROFILE_DETAIL_ALL_LAYERS"] = "0"

import torch

from orchid.llmscheduler.model_params import infer_model_params
from orchid.llmscheduler.runtime.base import AttentionContext
from orchid.llmscheduler.runtime.trt_runtime import TensorRTModelRuntime


def _sum_event_ms(pairs):
    total = 0.0
    for a, b in pairs:
        try:
            total += float(a.elapsed_time(b))
        except Exception:
            pass
    return float(total)


def _bench_trt_profile(
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
    runtime = TensorRTModelRuntime(model_path, use_fp16=fp16, engine_path=engine_path)
    mp = infer_model_params(model_path, tokenizer_path)

    page_size = int(mp.page_size)
    num_layers = int(mp.num_layers)
    pages_per_req = int((int(prefill_len) + int(decode_steps) + int(page_size) - 1) // int(page_size))
    pages_per_layer_need = int(int(bs) * int(max(1, pages_per_req)))
    max_pages = int(max(int(mp.max_pages), int(pages_per_layer_need) * int(num_layers)))
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
    ctx._prof = defaultdict(float)
    ctx._prof_events = {}

    vocab = 151936
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        vocab = int(getattr(tok, "vocab_size", vocab)) or vocab
    except Exception:
        pass

    torch.manual_seed(42)
    prompt = torch.randint(low=0, high=vocab, size=(int(bs), int(prefill_len)), dtype=torch.int64, device="cpu")
    input_ids = prompt.reshape(-1).contiguous()

    ctx.current_batch_req_ids = list(range(int(bs)))
    ctx.current_batch_seq_lens = [int(prefill_len)] * int(bs)
    ctx.current_batch_total_lens = [int(prefill_len)] * int(bs)
    ctx.current_batch_history_lens = [0] * int(bs)
    ctx.current_batch_is_prefill = [True] * int(bs)
    ctx.is_all_decode = False

    _ = runtime.forward(input_ids.to(device).to(torch.int32), ctx)

    for i in range(int(warmup_steps)):
        tok = torch.randint(low=0, high=vocab, size=(int(bs),), dtype=torch.int64, device="cpu")
        ctx.current_batch_seq_lens = [1] * int(bs)
        ctx.current_batch_total_lens = [int(prefill_len) + i + 1] * int(bs)
        ctx.current_batch_history_lens = [int(prefill_len) + i] * int(bs)
        ctx.current_batch_is_prefill = [False] * int(bs)
        ctx.is_all_decode = True
        _ = runtime.forward(tok.to(device).to(torch.int32), ctx)

    times_ms = []
    for i in range(int(decode_steps)):
        tok = torch.randint(low=0, high=vocab, size=(int(bs),), dtype=torch.int64, device="cpu")
        ctx.current_batch_seq_lens = [1] * int(bs)
        ctx.current_batch_total_lens = [int(prefill_len) + int(warmup_steps) + i + 1] * int(bs)
        ctx.current_batch_history_lens = [int(prefill_len) + int(warmup_steps) + i] * int(bs)
        ctx.current_batch_is_prefill = [False] * int(bs)
        ctx.is_all_decode = True
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = runtime.forward(tok.to(device).to(torch.int32), ctx)
        torch.cuda.synchronize()
        times_ms.append((time.perf_counter() - t0) * 1000.0)

    mean_ms = float(statistics.mean(times_ms)) if times_ms else 0.0
    tok_s = (float(bs) * float(decode_steps)) / (float(sum(times_ms)) / 1000.0) if times_ms else 0.0

    torch.cuda.synchronize()
    events = dict(getattr(ctx, "_prof_events", {}) or {})
    events_ms = {k: _sum_event_ms(v) for k, v in events.items()}

    prof = dict(getattr(ctx, "_prof", {}) or {})
    return {
        "decode_step_ms_mean": mean_ms,
        "decode_tok_s": tok_s,
        "prof_ms": prof,
        "prof_events_ms": events_ms,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--engine", default="")
    ap.add_argument("--bs", type=int, default=10)
    ap.add_argument("--prefill-len", type=int, default=128)
    ap.add_argument("--decode-steps", type=int, default=128)
    ap.add_argument("--warmup-steps", type=int, default=10)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    engine_path = str(args.engine).strip() or None
    out = _bench_trt_profile(
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
    print(out, flush=True)


if __name__ == "__main__":
    main()
