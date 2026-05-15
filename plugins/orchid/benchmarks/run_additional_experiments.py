import argparse
import csv
import os
from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import torch

from orchid.llmscheduler.model_params import infer_model_params
from orchid.llmscheduler.runtime.base import AttentionContext
from orchid.llmscheduler.runtime.trt_runtime import TensorRTModelRuntime
from orchid.paths import benchmark_artifact

from run_simple_suite import (
    _bench_trt_decode_tok_s,
    _bench_vllm_decode_tok_s,
    _trt_extract_last_logits,
    _trt_prefill_logits,
    _trt_teacher_forced_next_token_match,
    _vllm_greedy_generate,
)


@dataclass(frozen=True)
class Scenario:
    name: str
    bs: int
    prefill_len: int
    decode_steps: int


def _ensure_dir(path: str) -> None:
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)


def _tokenizer_vocab(tokenizer_path: str) -> int:
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        v = int(getattr(tok, "vocab_size", 0) or 0)
        return v if v > 0 else 151936
    except Exception:
        return 151936


def _prompt(seed: int, *, bs: int, prefill_len: int, vocab: int) -> torch.Tensor:
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))
    return torch.randint(low=0, high=int(vocab), size=(int(bs), int(prefill_len)), dtype=torch.int64, device="cpu", generator=g)


def _make_ctx_small(mp, *, bs: int, prefill_len: int, decode_steps: int, device: str, fp16: bool) -> AttentionContext:
    page_size = int(mp.page_size)
    pages_per_req = int((int(prefill_len) + int(decode_steps) + int(page_size) - 1) // int(page_size))
    pages_per_layer_need = int(int(bs) * int(max(1, pages_per_req)))
    max_pages = int(pages_per_layer_need) * int(mp.num_layers) * 2
    max_pages = max(max_pages, int(mp.num_layers) * 64)
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


def _plot_save(fig, path: str) -> str:
    _ensure_dir(path)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _write_csv(rows: list[dict[str, Any]], path: str) -> str:
    _ensure_dir(path)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["empty"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return path


def run_bs4_p64_s128_distribution(runtime, llm, mp, *, fp16: bool, device: str, vocab: int, out_dir: str, num_seeds: int) -> dict[str, str]:
    sc = Scenario("bs4_p64_s128", 4, 64, 128)
    rows: list[dict[str, Any]] = []
    for seed in range(int(num_seeds)):
        ctx = _make_ctx_small(mp, bs=sc.bs, prefill_len=sc.prefill_len, decode_steps=sc.decode_steps, device=device, fp16=fp16)
        prompt = _prompt(seed, bs=sc.bs, prefill_len=sc.prefill_len, vocab=vocab)
        vllm_tokens = _vllm_greedy_generate(llm, prompt.tolist(), decode_steps=sc.decode_steps)

        prefill_logits = _trt_prefill_logits(runtime, ctx, prompt)
        last_logits = _trt_extract_last_logits(prefill_logits, bs=sc.bs, prefill_len=sc.prefill_len)
        tok = torch.argmax(last_logits, dim=-1).to(ctx.device, dtype=torch.int32)
        trt_tokens: list[list[int]] = [[int(x)] for x in tok.detach().cpu().tolist()]
        history = [int(sc.prefill_len)] * int(sc.bs)
        total = [int(sc.prefill_len) + 1] * int(sc.bs)
        ctx.current_batch_req_ids = list(range(int(sc.bs)))
        for step in range(1, int(sc.decode_steps)):
            ctx.current_batch_seq_lens = [1] * int(sc.bs)
            ctx.current_batch_history_lens = list(history)
            ctx.current_batch_total_lens = list(total)
            ctx.current_batch_is_prefill = [False] * int(sc.bs)
            ctx.is_all_decode = True
            logits = runtime.forward(tok, ctx)
            tok = torch.argmax(logits, dim=-1).to(ctx.device, dtype=torch.int32)
            ids = tok.detach().cpu().tolist()
            for i in range(int(sc.bs)):
                trt_tokens[i].append(int(ids[i]))
            history = [int(h) + 1 for h in history]
            total = [int(t) + 1 for t in total]

        match = 0
        total = int(sc.bs) * int(sc.decode_steps)
        prefix_lens = []
        for i in range(int(sc.bs)):
            p = 0
            for t in range(int(sc.decode_steps)):
                if int(trt_tokens[i][t]) != int(vllm_tokens[i][t]):
                    break
                p += 1
            prefix_lens.append(int(p))
            for t in range(int(sc.decode_steps)):
                if int(trt_tokens[i][t]) == int(vllm_tokens[i][t]):
                    match += 1
        token_match_rate = float(match) / float(max(1, total))

        tf = _trt_teacher_forced_next_token_match(runtime, ctx, prompt, vllm_tokens, decode_steps=sc.decode_steps)

        rows.append(
            {
                "seed": int(seed),
                "token_match_rate": float(token_match_rate),
                "prefix_match_mean": float(sum(prefix_lens) / max(1, len(prefix_lens))),
                "prefix_match_min": int(min(prefix_lens)) if prefix_lens else 0,
                "prefix_match_max": int(max(prefix_lens)) if prefix_lens else 0,
                **tf,
            }
        )
        ctx.close()
        del ctx
        torch.cuda.empty_cache()

    csv_path = os.path.join(out_dir, "bs4_p64_s128_seed_sweep.csv")
    _write_csv(rows, csv_path)

    xs = [r["seed"] for r in rows]
    y_tm = [r["token_match_rate"] for r in rows]
    y_tf = [r["tf_next_token_match_rate"] for r in rows]
    y_pmin = [r["prefix_match_min"] for r in rows]

    fig = plt.figure(figsize=(10, 6))
    ax1 = fig.add_subplot(2, 1, 1)
    ax1.plot(xs, y_tm, marker="o", label="token_match_rate (free-run)")
    ax1.plot(xs, y_tf, marker="s", label="tf_next_token_match_rate")
    ax1.set_ylim(0.0, 1.02)
    ax1.set_ylabel("match rate")
    ax1.grid(True, axis="y", alpha=0.2)
    ax1.legend()

    ax2 = fig.add_subplot(2, 1, 2)
    ax2.bar(xs, y_pmin)
    ax2.set_xlabel("seed")
    ax2.set_ylabel("min prefix length")
    ax2.grid(True, axis="y", alpha=0.2)

    png_path = os.path.join(out_dir, "bs4_p64_s128_seed_sweep.png")
    _plot_save(fig, png_path)

    return {"csv": csv_path, "png": png_path}


def run_prefill_only_sweep(runtime, llm, mp, *, fp16: bool, device: str, vocab: int, out_dir: str, bs: int, prefill_lens: list[int], num_seeds: int) -> dict[str, str]:
    rows: list[dict[str, Any]] = []
    for L in prefill_lens:
        match = 0
        total = int(bs) * int(num_seeds)
        for seed in range(int(num_seeds)):
            ctx = _make_ctx_small(mp, bs=bs, prefill_len=int(L), decode_steps=1, device=device, fp16=fp16)
            prompt = _prompt(seed, bs=bs, prefill_len=int(L), vocab=vocab)
            vllm_tokens = _vllm_greedy_generate(llm, prompt.tolist(), decode_steps=1)
            prefill_logits = _trt_prefill_logits(runtime, ctx, prompt)
            last_logits = _trt_extract_last_logits(prefill_logits, bs=bs, prefill_len=int(L))
            trt_next = torch.argmax(last_logits, dim=-1).detach().cpu().tolist()
            for i in range(int(bs)):
                if int(trt_next[i]) == int(vllm_tokens[i][0]):
                    match += 1
            ctx.close()
            del ctx
            torch.cuda.empty_cache()
        rows.append({"prefill_len": int(L), "bs": int(bs), "seeds": int(num_seeds), "prefill_next_token_match_rate": float(match) / float(max(1, total))})

    csv_path = os.path.join(out_dir, "prefill_only_sweep.csv")
    _write_csv(rows, csv_path)

    fig = plt.figure(figsize=(7, 4))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot([r["prefill_len"] for r in rows], [r["prefill_next_token_match_rate"] for r in rows], marker="o")
    ax.set_xlabel("prefill_len")
    ax.set_ylabel("prefill next-token match")
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, axis="y", alpha=0.2)
    png_path = os.path.join(out_dir, "prefill_only_sweep.png")
    _plot_save(fig, png_path)
    return {"csv": csv_path, "png": png_path}


def run_perf_variance(
    runtime, llm_eager, llm_graph, mp, *, fp16: bool, device: str, vocab: int, out_dir: str, repeats: int
) -> dict[str, str]:
    sc = Scenario("bs10_p128_s128", 10, 128, 128)
    ctx = _make_ctx_small(mp, bs=sc.bs, prefill_len=sc.prefill_len, decode_steps=sc.decode_steps, device=device, fp16=fp16)
    prompt = _prompt(0, bs=sc.bs, prefill_len=sc.prefill_len, vocab=vocab)

    trt = []
    trt_cg = []
    vllm_e = []
    vllm_g = []
    for _ in range(int(repeats)):
        os.environ["LLMSCHEDULER_TRT_CUDAGRAPH"] = "0"
        trt_perf = _bench_trt_decode_tok_s(runtime, ctx, prompt, decode_steps=sc.decode_steps, warmup_steps=10, vocab=vocab)
        trt.append(float(trt_perf["decode_tok_s"]))
        os.environ["LLMSCHEDULER_TRT_CUDAGRAPH"] = "1"
        trt_perf_cg = _bench_trt_decode_tok_s(
            runtime, ctx, prompt, decode_steps=sc.decode_steps, warmup_steps=10, vocab=vocab, freeze_lens=True
        )
        trt_cg.append(float(trt_perf_cg["decode_tok_s"]))
        os.environ["LLMSCHEDULER_TRT_CUDAGRAPH"] = "0"
        vllm_perf = _bench_vllm_decode_tok_s(llm_eager, prompt.tolist(), decode_steps=sc.decode_steps)
        vllm_e.append(float(vllm_perf["decode_tok_s"]))
        if llm_graph is not None:
            vllm_perf_g = _bench_vllm_decode_tok_s(llm_graph, prompt.tolist(), decode_steps=sc.decode_steps)
            vllm_g.append(float(vllm_perf_g["decode_tok_s"]))

    def stat(xs):
        n = len(xs)
        mu = sum(xs) / max(1, n)
        var = sum((x - mu) ** 2 for x in xs) / max(1, n - 1) if n > 1 else 0.0
        sd = var**0.5
        cv = (sd / mu) if mu > 0 else 0.0
        return mu, sd, cv

    trt_mu, trt_sd, trt_cv = stat(trt)
    trt_cg_mu, trt_cg_sd, trt_cg_cv = stat(trt_cg)
    ve_mu, ve_sd, ve_cv = stat(vllm_e)
    vg_mu, vg_sd, vg_cv = stat(vllm_g) if vllm_g else (0.0, 0.0, 0.0)

    rows = [
        {"engine": "TRT", "scenario": sc.name, "repeats": int(repeats), "mean_tok_s": trt_mu, "std_tok_s": trt_sd, "cv": trt_cv},
        {"engine": "TRT cg", "scenario": sc.name, "repeats": int(repeats), "mean_tok_s": trt_cg_mu, "std_tok_s": trt_cg_sd, "cv": trt_cg_cv},
        {"engine": "vLLM eager", "scenario": sc.name, "repeats": int(repeats), "mean_tok_s": ve_mu, "std_tok_s": ve_sd, "cv": ve_cv},
    ]
    if llm_graph is not None:
        rows.append({"engine": "vLLM graph", "scenario": sc.name, "repeats": int(repeats), "mean_tok_s": vg_mu, "std_tok_s": vg_sd, "cv": vg_cv})
    csv_path = os.path.join(out_dir, "perf_variance.csv")
    _write_csv(rows, csv_path)

    fig = plt.figure(figsize=(7, 4))
    ax = fig.add_subplot(1, 1, 1)
    names = [r["engine"] for r in rows]
    means = [float(r["mean_tok_s"]) for r in rows]
    stds = [float(r["std_tok_s"]) for r in rows]
    ax.bar(names, means, yerr=stds, capsize=5)
    ax.set_ylabel("decode tok/s (mean ± std)")
    ax.grid(True, axis="y", alpha=0.2)
    png_path = os.path.join(out_dir, "perf_variance.png")
    _plot_save(fig, png_path)
    return {"csv": csv_path, "png": png_path}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--engine", default="")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out-dir", default=benchmark_artifact())
    ap.add_argument("--vllm-gpu-mem", type=float, default=0.25)
    ap.add_argument("--seed-sweep", type=int, default=20)
    ap.add_argument("--prefill-seeds", type=int, default=30)
    ap.add_argument("--perf-repeats", type=int, default=3)
    ap.add_argument("--vllm-graph-perf", action="store_true")
    args = ap.parse_args()

    out_dir = str(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    os.environ.setdefault("LLMSCHEDULER_FLASHINFER_WORKSPACE_MB", "64")

    engine_path = str(args.engine).strip() or None
    vocab = _tokenizer_vocab(args.tokenizer)

    mp = infer_model_params(args.model, args.tokenizer)
    runtime = TensorRTModelRuntime(args.model, use_fp16=bool(args.fp16), engine_path=engine_path)

    from vllm import LLM

    dtype = "float16" if bool(args.fp16) else "float32"
    max_model_len = 1024
    llm = LLM(
        model=args.tokenizer,
        tokenizer=args.tokenizer,
        dtype=dtype,
        trust_remote_code=True,
        max_model_len=int(max_model_len),
        gpu_memory_utilization=float(args.vllm_gpu_mem),
        disable_log_stats=True,
        enforce_eager=True,
    )
    llm_graph = None
    if bool(args.vllm_graph_perf):
        llm_graph = LLM(
            model=args.tokenizer,
            tokenizer=args.tokenizer,
            dtype=dtype,
            trust_remote_code=True,
            max_model_len=int(max_model_len),
            gpu_memory_utilization=float(args.vllm_gpu_mem),
            disable_log_stats=True,
            enforce_eager=False,
        )

    a = run_bs4_p64_s128_distribution(runtime, llm, mp, fp16=bool(args.fp16), device=str(args.device), vocab=vocab, out_dir=out_dir, num_seeds=int(args.seed_sweep))
    b = run_prefill_only_sweep(
        runtime,
        llm,
        mp,
        fp16=bool(args.fp16),
        device=str(args.device),
        vocab=vocab,
        out_dir=out_dir,
        bs=4,
        prefill_lens=[16, 32, 64, 128, 256, 512],
        num_seeds=int(args.prefill_seeds),
    )
    c = run_perf_variance(runtime, llm, llm_graph, mp, fp16=bool(args.fp16), device=str(args.device), vocab=vocab, out_dir=out_dir, repeats=int(args.perf_repeats))

    print({"seed_sweep": a, "prefill_only": b, "perf_variance": c}, flush=True)


if __name__ == "__main__":
    main()
