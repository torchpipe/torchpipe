import argparse
import csv
import os
import statistics
import time

import torch

from orchid.llmscheduler.model_params import infer_model_params
from orchid.llmscheduler.runtime.base import AttentionContext
from orchid.llmscheduler.runtime.trt_runtime import TensorRTModelRuntime
from orchid.paths import test_artifact


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
    vocab: int,
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
    return {"decode_step_ms_mean": mean_ms, "decode_tok_s": tok_s}


def _ensure_dir(p: str):
    d = os.path.dirname(os.path.abspath(p))
    if d:
        os.makedirs(d, exist_ok=True)


def _bench_vllm_matrix(llm, prompt_token_ids_list, decode_steps: int):
    from vllm import SamplingParams

    sampling = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=int(decode_steps))
    prompts = [{"prompt_token_ids": ids} for ids in prompt_token_ids_list]
    t0 = time.perf_counter()
    _ = llm.generate(prompts, sampling)
    dt = float(time.perf_counter() - t0)
    return dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--engine", default="")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--decode-steps", type=int, default=128)
    ap.add_argument("--warmup-steps", type=int, default=10)
    ap.add_argument("--bs-list", default="1,4,8,16,32")
    ap.add_argument("--prefill-list", default="128,512,2048")
    ap.add_argument("--out-csv", default=test_artifact("perf_matrix.csv"))
    ap.add_argument("--skip-vllm", action="store_true")
    ap.add_argument("--vllm-gpu-mem", type=float, default=0.4)
    ap.add_argument("--vllm-eager", action="store_true")
    args = ap.parse_args()

    engine_path = str(args.engine).strip() or None
    bs_list = [int(x.strip()) for x in str(args.bs_list).split(",") if x.strip()]
    prefill_list = [int(x.strip()) for x in str(args.prefill_list).split(",") if x.strip()]

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    vocab = int(getattr(tokenizer, "vocab_size", 151936)) or 151936

    llm = None
    if not args.skip_vllm:
        from vllm import LLM

        dtype = "float16" if bool(args.fp16) else "float32"
        max_model_len = int(max(prefill_list) + int(args.decode_steps) + 32)
        llm = LLM(
            model=args.tokenizer,
            tokenizer=args.tokenizer,
            dtype=dtype,
            trust_remote_code=True,
            max_model_len=max_model_len,
            gpu_memory_utilization=float(args.vllm_gpu_mem),
            disable_log_stats=True,
            enforce_eager=bool(args.vllm_eager),
        )
        _ = _bench_vllm_matrix(llm, [[0, 1, 2, 3]], decode_steps=1)

    _ensure_dir(args.out_csv)
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "bs",
                "prefill_len",
                "trt_decode_tok_s",
                "trt_decode_step_ms_mean",
                "vllm_decode_tok_s",
                "vllm_warm_s",
                "vllm_cached_s",
                "error",
            ],
        )
        w.writeheader()
        for bs in bs_list:
            for prefill_len in prefill_list:
                row = {
                    "bs": int(bs),
                    "prefill_len": int(prefill_len),
                    "trt_decode_tok_s": "",
                    "trt_decode_step_ms_mean": "",
                    "vllm_decode_tok_s": "",
                    "vllm_warm_s": "",
                    "vllm_cached_s": "",
                    "error": "",
                }
                try:
                    trt = _bench_trt(
                        args.model,
                        args.tokenizer,
                        engine_path=engine_path,
                        fp16=bool(args.fp16),
                        bs=int(bs),
                        prefill_len=int(prefill_len),
                        decode_steps=int(args.decode_steps),
                        warmup_steps=int(args.warmup_steps),
                        device=str(args.device),
                        vocab=int(vocab),
                    )
                    row["trt_decode_tok_s"] = float(trt["decode_tok_s"])
                    row["trt_decode_step_ms_mean"] = float(trt["decode_step_ms_mean"])
                except Exception as e:
                    row["error"] = f"trt:{e}"

                if llm is not None:
                    try:
                        torch.manual_seed(123)
                        prompt = torch.randint(
                            low=0, high=int(vocab), size=(int(bs), int(prefill_len)), dtype=torch.int64, device="cpu"
                        )
                        ids_list = prompt.tolist()
                        dt0 = _bench_vllm_matrix(llm, ids_list, decode_steps=int(args.decode_steps))
                        dt1 = _bench_vllm_matrix(llm, ids_list, decode_steps=int(args.decode_steps))
                        row["vllm_warm_s"] = float(dt0)
                        row["vllm_cached_s"] = float(dt1)
                        row["vllm_decode_tok_s"] = (float(bs) * float(args.decode_steps)) / float(dt1) if dt1 > 0 else 0.0
                    except Exception as e:
                        row["error"] = (row["error"] + " | " if row["error"] else "") + f"vllm:{e}"

                w.writerow(row)
                f.flush()
                print(row, flush=True)


if __name__ == "__main__":
    main()
