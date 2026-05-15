import argparse
import csv
import os
import time
from dataclasses import dataclass
from typing import Any
import zlib
import subprocess
import sys
import tempfile

import torch

from orchid.llmscheduler.model_params import infer_model_params
from orchid.llmscheduler.runtime.base import AttentionContext
from orchid.llmscheduler.runtime.trt_runtime import TensorRTModelRuntime
from orchid.paths import benchmark_artifact


@dataclass(frozen=True)
class Scenario:
    name: str
    bs: int
    prefill_len: int
    decode_steps: int


def _preset_scenarios(preset: str) -> list[Scenario]:
    p = str(preset or "").strip().lower()
    if p in ("basic", "default", ""):
        return [
            Scenario("bs1_p16_s8", 1, 16, 8),
            Scenario("bs4_p128_s32", 4, 128, 32),
            Scenario("bs10_p128_s128", 10, 128, 128),
        ]
    if p in ("more", "simple_more", "simple-more"):
        return [
            Scenario("bs1_p16_s8", 1, 16, 8),
            Scenario("bs1_p128_s64", 1, 128, 64),
            Scenario("bs2_p128_s128", 2, 128, 128),
            Scenario("bs4_p64_s128", 4, 64, 128),
            Scenario("bs4_p128_s32", 4, 128, 32),
            Scenario("bs8_p128_s128", 8, 128, 128),
            Scenario("bs10_p128_s128", 10, 128, 128),
            Scenario("bs16_p128_s64", 16, 128, 64),
        ]
    if p in ("longprefill", "long_prefill", "long-prefill"):
        return [
            Scenario("bs1_p512_s32", 1, 512, 32),
            Scenario("bs4_p512_s32", 4, 512, 32),
            Scenario("bs8_p512_s64", 8, 512, 64),
        ]
    raise ValueError(f"Unknown preset: {preset}")


def _parse_scenarios(spec: str) -> list[Scenario]:
    s = str(spec or "").strip()
    if not s:
        return []
    out: list[Scenario] = []
    for chunk in s.split(";"):
        c = chunk.strip()
        if not c:
            continue
        parts = [x.strip() for x in c.split(",")]
        if len(parts) == 4:
            name, bs, p, steps = parts
        elif len(parts) == 3:
            bs, p, steps = parts
            name = f"bs{bs}_p{p}_s{steps}"
        else:
            raise ValueError(f"Bad scenario spec chunk: {c}")
        out.append(Scenario(str(name), int(bs), int(p), int(steps)))
    return out


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


def _trt_greedy_generate(
    runtime: TensorRTModelRuntime,
    ctx: AttentionContext,
    prompt_cpu_i64: torch.Tensor,
    *,
    decode_steps: int,
) -> list[list[int]]:
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


def _trt_teacher_forced_next_token_match(
    runtime: TensorRTModelRuntime,
    ctx: AttentionContext,
    prompt_cpu_i64: torch.Tensor,
    vllm_tokens: list[list[int]],
    *,
    decode_steps: int,
) -> dict[str, float]:
    bs, prefill_len = int(prompt_cpu_i64.shape[0]), int(prompt_cpu_i64.shape[1])
    if len(vllm_tokens) != bs:
        raise RuntimeError(f"vllm_tokens bs mismatch: {len(vllm_tokens)} vs {bs}")
    for i in range(bs):
        if len(vllm_tokens[i]) < int(decode_steps):
            raise RuntimeError(f"vllm_tokens[{i}] too short: {len(vllm_tokens[i])} < {decode_steps}")

    prefill_logits = _trt_prefill_logits(runtime, ctx, prompt_cpu_i64)
    last_logits = _trt_extract_last_logits(prefill_logits, bs=bs, prefill_len=prefill_len)
    pred0 = torch.argmax(last_logits, dim=-1).detach().cpu().tolist()
    match = 0
    total = bs * max(0, int(decode_steps) - 0)
    prefix_lens = [0 for _ in range(bs)]

    for i in range(bs):
        if int(pred0[i]) == int(vllm_tokens[i][0]):
            match += 1
            prefix_lens[i] += 1

    ctx.current_batch_req_ids = list(range(bs))
    tok = torch.tensor([int(vllm_tokens[i][0]) for i in range(bs)], dtype=torch.int32, device=ctx.device)

    for t in range(0, int(decode_steps) - 1):
        ctx.current_batch_seq_lens = [1] * bs
        ctx.current_batch_history_lens = [int(prefill_len) + int(t)] * bs
        ctx.current_batch_total_lens = [int(prefill_len) + int(t) + 1] * bs
        ctx.current_batch_is_prefill = [False] * bs
        ctx.is_all_decode = True
        logits = runtime.forward(tok, ctx)
        pred = torch.argmax(logits, dim=-1).detach().cpu().tolist()
        for i in range(bs):
            if prefix_lens[i] == int(t) + 1 and int(pred[i]) == int(vllm_tokens[i][t + 1]):
                prefix_lens[i] += 1
            if int(pred[i]) == int(vllm_tokens[i][t + 1]):
                match += 1
        tok = torch.tensor([int(vllm_tokens[i][t + 1]) for i in range(bs)], dtype=torch.int32, device=ctx.device)

    rate = float(match) / float(max(1, total))
    return {
        "tf_next_token_match_rate": float(rate),
        "tf_prefix_match_mean": float(sum(prefix_lens) / float(max(1, len(prefix_lens)))),
        "tf_prefix_match_min": float(min(prefix_lens) if prefix_lens else 0),
        "tf_prefix_match_max": float(max(prefix_lens) if prefix_lens else 0),
    }


def _vllm_greedy_generate(llm, prompts_token_ids: list[list[int]], *, decode_steps: int) -> list[list[int]]:
    from vllm import SamplingParams

    sampling = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=int(decode_steps), ignore_eos=True)
    outputs = llm.generate([{"prompt_token_ids": ids} for ids in prompts_token_ids], sampling)
    out: list[list[int]] = []
    for o in outputs:
        out.append([int(x) for x in o.outputs[0].token_ids])
    return out


def _bench_trt_decode_tok_s(
    runtime: TensorRTModelRuntime,
    ctx: AttentionContext,
    prompt_cpu_i64: torch.Tensor,
    *,
    decode_steps: int,
    warmup_steps: int,
    vocab: int,
    freeze_lens: bool = False,
) -> dict[str, float]:
    bs, prefill_len = int(prompt_cpu_i64.shape[0]), int(prompt_cpu_i64.shape[1])
    prev_cg = os.environ.get("LLMSCHEDULER_TRT_CUDAGRAPH", "0")
    os.environ["LLMSCHEDULER_TRT_CUDAGRAPH"] = "0"
    _ = _trt_prefill_logits(runtime, ctx, prompt_cpu_i64)
    os.environ["LLMSCHEDULER_TRT_CUDAGRAPH"] = str(prev_cg)

    tok = torch.randint(low=0, high=int(vocab), size=(int(bs),), dtype=torch.int32, device=ctx.device)
    history = [int(prefill_len)] * int(bs)
    total = [int(prefill_len) + 1] * int(bs)

    for i in range(int(warmup_steps)):
        ctx.current_batch_seq_lens = [1] * int(bs)
        ctx.current_batch_history_lens = list(history)
        ctx.current_batch_total_lens = list(total)
        ctx.current_batch_is_prefill = [False] * int(bs)
        ctx.is_all_decode = True
        _ = runtime.forward(tok, ctx)
        if not bool(freeze_lens):
            history = [int(h) + 1 for h in history]
            total = [int(t) + 1 for t in total]

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for i in range(int(decode_steps)):
        ctx.current_batch_seq_lens = [1] * int(bs)
        ctx.current_batch_history_lens = list(history)
        ctx.current_batch_total_lens = list(total)
        ctx.current_batch_is_prefill = [False] * int(bs)
        ctx.is_all_decode = True
        _ = runtime.forward(tok, ctx)
        if not bool(freeze_lens):
            history = [int(h) + 1 for h in history]
            total = [int(t) + 1 for t in total]
    torch.cuda.synchronize()
    dt = float(time.perf_counter() - t0)
    tok_s_e2e = (float(bs) * float(decode_steps)) / float(dt) if dt > 0 else 0.0

    times = []
    for i in range(int(decode_steps)):
        ctx.current_batch_seq_lens = [1] * int(bs)
        ctx.current_batch_history_lens = list(history)
        ctx.current_batch_total_lens = list(total)
        ctx.current_batch_is_prefill = [False] * int(bs)
        ctx.is_all_decode = True
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = runtime.forward(tok, ctx)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)
        if not bool(freeze_lens):
            history = [int(h) + 1 for h in history]
            total = [int(t) + 1 for t in total]

    mean_ms = float(sum(times) / max(1, len(times)))
    tok_s_step = (float(bs) * float(decode_steps)) / (float(sum(times)) / 1000.0) if times else 0.0
    return {
        "decode_step_ms_mean": float(mean_ms),
        "decode_tok_s": float(tok_s_e2e),
        "decode_tok_s_e2e": float(tok_s_e2e),
        "decode_tok_s_step_sync": float(tok_s_step),
    }


def _bench_vllm_decode_tok_s(llm, prompt_token_ids_list: list[list[int]], *, decode_steps: int) -> dict[str, float]:
    from vllm import SamplingParams

    sampling = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=int(decode_steps), ignore_eos=True)
    prompts = [{"prompt_token_ids": ids} for ids in prompt_token_ids_list]
    _ = llm.generate([{"prompt_token_ids": [0, 1, 2, 3]}], SamplingParams(temperature=0.0, top_p=1.0, max_tokens=1, ignore_eos=True))
    t0 = time.perf_counter()
    _ = llm.generate(prompts, sampling)
    dt0 = float(time.perf_counter() - t0)
    t1 = time.perf_counter()
    _ = llm.generate(prompts, sampling)
    dt1 = float(time.perf_counter() - t1)
    bs = int(len(prompt_token_ids_list))
    tok_s_cached = (float(bs) * float(decode_steps)) / float(dt1) if dt1 > 0 else 0.0
    return {"warm_s": float(dt0), "cached_s": float(dt1), "decode_tok_s": float(tok_s_cached)}


def _maybe_plot(csv_path: str, png_path: str) -> str | None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return None

    rows = []
    with open(csv_path, "r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    if not rows:
        return None

    names = [row["scenario"] for row in rows]
    trt = [float(row.get("trt_decode_tok_s") or 0.0) for row in rows]
    trt_cg = [float(row.get("trt_decode_tok_s_cg") or 0.0) for row in rows]
    vllm_eager = [float(row.get("vllm_decode_tok_s") or 0.0) for row in rows]
    vllm_graph = [float(row.get("vllm_decode_tok_s_graph") or 0.0) for row in rows]
    ratio_eager = [(t / v if v > 0 else 0.0) for t, v in zip(trt, vllm_eager)]
    ratio_graph = [(t / v if v > 0 else 0.0) for t, v in zip(trt, vllm_graph)]
    ratio_trt_cg = [(t / v if v > 0 else 0.0) for t, v in zip(trt_cg, vllm_graph)]
    match = [float(row.get("token_match_rate") or 0.0) for row in rows]

    x = list(range(len(names)))
    w = 0.22
    plt.figure(figsize=(max(8, len(names) * 1.35), 6))
    plt.subplot(2, 1, 1)
    plt.bar([i - 1.5 * w for i in x], trt, width=w, label="TRT")
    if any(v > 0 for v in trt_cg):
        plt.bar([i - 0.5 * w for i in x], trt_cg, width=w, label="TRT cg")
    plt.bar([i + 0.5 * w for i in x], vllm_eager, width=w, label="vLLM eager")
    if any(v > 0 for v in vllm_graph):
        plt.bar([i + 1.5 * w for i in x], vllm_graph, width=w, label="vLLM graph")
    plt.xticks(x, names, rotation=20, ha="right")
    plt.ylabel("decode tok/s (cached)")
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.plot(x, ratio_eager, marker="o", label="TRT/vLLM eager")
    if any(v > 0 for v in vllm_graph):
        plt.plot(x, ratio_graph, marker="^", label="TRT/vLLM graph")
    if any(v > 0 for v in trt_cg) and any(v > 0 for v in vllm_graph):
        plt.plot(x, ratio_trt_cg, marker="x", label="TRT cg/vLLM graph")
    plt.plot(x, match, marker="s", label="token match")
    plt.xticks(x, names, rotation=20, ha="right")
    plt.ylim(0.0, 1.05)
    plt.ylabel("ratio / match")
    plt.grid(True, axis="y", alpha=0.2)
    plt.legend()
    plt.tight_layout()
    _ensure_dir(png_path)
    plt.savefig(png_path, dpi=160)
    return png_path


def _write_summary(rows: list[dict[str, Any]], md_path: str) -> str:
    _ensure_dir(md_path)
    has_vllm = any((r.get("vllm_decode_tok_s") not in ("", None) and float(r.get("vllm_decode_tok_s") or 0.0) > 0) for r in rows)
    has_vllm_graph = any(
        (r.get("vllm_decode_tok_s_graph") not in ("", None) and float(r.get("vllm_decode_tok_s_graph") or 0.0) > 0) for r in rows
    )
    has_trt_cg = any((r.get("trt_decode_tok_s_cg") not in ("", None) and float(r.get("trt_decode_tok_s_cg") or 0.0) > 0) for r in rows)
    ratios = []
    ratios_graph = []
    ratios_trt_cg = []
    for r in rows:
        vt = float(r.get("vllm_decode_tok_s") or 0.0)
        tt = float(r.get("trt_decode_tok_s") or 0.0)
        if vt > 0:
            ratios.append(tt / vt)
        vg = float(r.get("vllm_decode_tok_s_graph") or 0.0)
        if vg > 0:
            ratios_graph.append(tt / vg)
        tcg = float(r.get("trt_decode_tok_s_cg") or 0.0)
        if vg > 0 and tcg > 0:
            ratios_trt_cg.append(tcg / vg)
    avg_ratio = float(sum(ratios) / max(1, len(ratios))) if ratios else 0.0
    min_ratio = float(min(ratios)) if ratios else 0.0
    max_ratio = float(max(ratios)) if ratios else 0.0
    avg_ratio_graph = float(sum(ratios_graph) / max(1, len(ratios_graph))) if ratios_graph else 0.0
    min_ratio_graph = float(min(ratios_graph)) if ratios_graph else 0.0
    max_ratio_graph = float(max(ratios_graph)) if ratios_graph else 0.0
    avg_ratio_trt_cg = float(sum(ratios_trt_cg) / max(1, len(ratios_trt_cg))) if ratios_trt_cg else 0.0
    min_ratio_trt_cg = float(min(ratios_trt_cg)) if ratios_trt_cg else 0.0
    max_ratio_trt_cg = float(max(ratios_trt_cg)) if ratios_trt_cg else 0.0
    with open(md_path, "w") as f:
        f.write("# orchid simple suite summary\n\n")
        if has_vllm:
            f.write(f"- avg TRT/vLLM eager: {avg_ratio:.4f}\n")
            f.write(f"- min TRT/vLLM eager: {min_ratio:.4f}\n")
            f.write(f"- max TRT/vLLM eager: {max_ratio:.4f}\n")
        if has_vllm_graph:
            f.write(f"- avg TRT/vLLM graph: {avg_ratio_graph:.4f}\n")
            f.write(f"- min TRT/vLLM graph: {min_ratio_graph:.4f}\n")
            f.write(f"- max TRT/vLLM graph: {max_ratio_graph:.4f}\n")
        if has_trt_cg and has_vllm_graph:
            f.write(f"- avg TRT_cg/vLLM graph: {avg_ratio_trt_cg:.4f}\n")
            f.write(f"- min TRT_cg/vLLM graph: {min_ratio_trt_cg:.4f}\n")
            f.write(f"- max TRT_cg/vLLM graph: {max_ratio_trt_cg:.4f}\n")
        f.write("\n")
        f.write("|scenario|bs|prefill|steps|trt tok/s|trt cg tok/s|vllm eager tok/s|vllm graph tok/s|TRT/vllm eager|TRT/vllm graph|TRTcg/vllm graph|token match|prefix mean|min prefix|\n")
        f.write("|-:|-:|-:|-:|-:|-:|-:|-:|-:|-:|-:|-:|-:|-:|\n")
        for r in rows:
            vt = float(r.get("vllm_decode_tok_s") or 0.0)
            tt = float(r.get("trt_decode_tok_s") or 0.0)
            vg = float(r.get("vllm_decode_tok_s_graph") or 0.0)
            tcg = float(r.get("trt_decode_tok_s_cg") or 0.0)
            ratio = (tt / vt) if vt > 0 else 0.0
            ratio_g = (tt / vg) if vg > 0 else 0.0
            ratio_tcg = (tcg / vg) if vg > 0 and tcg > 0 else 0.0
            tm = r.get("token_match_rate")
            tmv = float(tm) if tm not in ("", None) else 0.0
            pmean = float(r.get("prefix_match_mean") or 0.0)
            pmin = int(float(r.get("prefix_match_min") or 0.0))
            f.write(
                f"|{r['scenario']}|{r['bs']}|{r['prefill_len']}|{r['decode_steps']}|{tt:.2f}|{(tcg if tcg > 0 else 0.0):.2f}|{(vt if vt > 0 else 0.0):.2f}|{(vg if vg > 0 else 0.0):.2f}|{ratio:.4f}|{ratio_g:.4f}|{ratio_tcg:.4f}|{tmv:.4f}|{pmean:.2f}|{pmin}|\n"
            )
    return md_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--engine", default="")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--warmup-steps", type=int, default=10)
    ap.add_argument("--out-csv", default=benchmark_artifact("simple_suite.csv"))
    ap.add_argument("--out-png", default=benchmark_artifact("simple_suite.png"))
    ap.add_argument("--out-md", default=benchmark_artifact("simple_suite.md"))
    ap.add_argument("--preset", default="more")
    ap.add_argument("--scenarios", default="")
    ap.add_argument("--teacher-force", action="store_true")
    ap.add_argument("--trt-cudagraph-perf", action="store_true")
    ap.add_argument("--skip-vllm", action="store_true")
    ap.add_argument("--vllm-eager-correctness", action="store_true")
    ap.add_argument("--vllm-graph-perf", action="store_true")
    ap.add_argument("--vllm-gpu-mem", type=float, default=0.25)
    args = ap.parse_args()

    os.environ.setdefault("LLMSCHEDULER_FLASHINFER_WORKSPACE_MB", "64")
    is_trt_cg_child = bool(int(os.environ.get("LLMSCHEDULER_TRT_CUDAGRAPH_CHILD", "0")))
    torch.manual_seed(int(args.seed))
    engine_path = str(args.engine).strip() or None
    vocab = _tokenizer_vocab(args.tokenizer)

    mp = infer_model_params(args.model, args.tokenizer)
    runtime = TensorRTModelRuntime(args.model, use_fp16=bool(args.fp16), engine_path=engine_path)

    scenarios = _parse_scenarios(str(args.scenarios)) or _preset_scenarios(str(args.preset))

    trt_cg_tok_s_by_name: dict[str, float] = {}
    if bool(args.trt_cudagraph_perf) and not bool(is_trt_cg_child):
        for sc in scenarios:
            with tempfile.TemporaryDirectory() as td:
                tmp_csv = os.path.join(td, "row.csv")
                tmp_md = os.path.join(td, "row.md")
                tmp_png = os.path.join(td, "row.png")
                env = dict(os.environ)
                env["LLMSCHEDULER_TRT_CUDAGRAPH_CHILD"] = "1"
                cmd = [
                    sys.executable,
                    os.path.abspath(__file__),
                    "--model",
                    str(args.model),
                    "--tokenizer",
                    str(args.tokenizer),
                    "--engine",
                    str(args.engine),
                    "--device",
                    str(args.device),
                    "--seed",
                    str(args.seed),
                    "--warmup-steps",
                    str(args.warmup_steps),
                    "--scenarios",
                    f"{sc.name},{int(sc.bs)},{int(sc.prefill_len)},{int(sc.decode_steps)}",
                    "--trt-cudagraph-perf",
                    "--skip-vllm",
                    "--out-csv",
                    tmp_csv,
                    "--out-md",
                    tmp_md,
                    "--out-png",
                    tmp_png,
                ]
                if bool(args.fp16):
                    cmd.append("--fp16")
                try:
                    _ = subprocess.check_output(cmd, env=env, stderr=subprocess.STDOUT, text=True)
                    with open(tmp_csv, "r", newline="") as f:
                        r = csv.DictReader(f)
                        row = next(iter(r), None)
                    if row is not None:
                        trt_cg_tok_s_by_name[str(sc.name)] = float(row.get("trt_decode_tok_s_cg") or 0.0)
                except subprocess.CalledProcessError:
                    trt_cg_tok_s_by_name[str(sc.name)] = 0.0

    llm = None
    llm_eager_perf = None
    llm_graph_perf = None
    if not bool(args.skip_vllm):
        from vllm import LLM

        dtype = "float16" if bool(args.fp16) else "float32"
        max_prefill = max(int(s.prefill_len) for s in scenarios)
        max_steps = max(int(s.decode_steps) for s in scenarios)
        llm = LLM(
            model=args.tokenizer,
            tokenizer=args.tokenizer,
            dtype=dtype,
            trust_remote_code=True,
            max_model_len=max(4096, int(max_prefill) + int(max_steps) + 32),
            gpu_memory_utilization=float(args.vllm_gpu_mem),
            disable_log_stats=True,
            enforce_eager=True,
        )
        llm_eager_perf = llm
        if bool(args.vllm_graph_perf):
            llm_graph_perf = LLM(
                model=args.tokenizer,
                tokenizer=args.tokenizer,
                dtype=dtype,
                trust_remote_code=True,
                max_model_len=max(4096, int(max_prefill) + int(max_steps) + 32),
                gpu_memory_utilization=float(args.vllm_gpu_mem),
                disable_log_stats=True,
                enforce_eager=False,
            )

    rows: list[dict[str, Any]] = []
    for sc in scenarios:
        ctx = _make_ctx(mp, bs=int(sc.bs), prefill_len=int(sc.prefill_len), decode_steps=int(sc.decode_steps), device=str(args.device), fp16=bool(args.fp16))
        g = torch.Generator(device="cpu")
        g.manual_seed(int(args.seed) ^ int(zlib.crc32(str(sc.name).encode("utf-8")) & 0xFFFFFFFF))
        prompt = torch.randint(
            low=0, high=int(vocab), size=(int(sc.bs), int(sc.prefill_len)), dtype=torch.int64, device="cpu", generator=g
        )

        os.environ["LLMSCHEDULER_TRT_CUDAGRAPH"] = "0"
        trt_tokens = _trt_greedy_generate(runtime, ctx, prompt, decode_steps=int(sc.decode_steps))
        trt_perf = _bench_trt_decode_tok_s(
            runtime, ctx, prompt, decode_steps=int(sc.decode_steps), warmup_steps=int(args.warmup_steps), vocab=int(vocab)
        )
        trt_perf_cg = {}
        if bool(args.trt_cudagraph_perf):
            if is_trt_cg_child:
                os.environ["LLMSCHEDULER_TRT_CUDAGRAPH"] = "1"
                trt_perf_cg = _bench_trt_decode_tok_s(
                    runtime,
                    ctx,
                    prompt,
                    decode_steps=int(sc.decode_steps),
                    warmup_steps=int(args.warmup_steps),
                    vocab=int(vocab),
                    freeze_lens=True,
                )
                os.environ["LLMSCHEDULER_TRT_CUDAGRAPH"] = "0"
            else:
                trt_perf_cg = {"decode_tok_s": float(trt_cg_tok_s_by_name.get(str(sc.name), 0.0))}

        vllm_acc = ""
        vllm_tok_s = ""
        vllm_warm_s = ""
        vllm_cached_s = ""
        vllm_tok_s_graph = ""
        vllm_warm_s_graph = ""
        vllm_cached_s_graph = ""
        if llm is not None:
            vllm_tokens = _vllm_greedy_generate(llm, prompt.tolist(), decode_steps=int(sc.decode_steps))
            match = 0
            total = int(sc.bs) * int(sc.decode_steps)
            for i in range(int(sc.bs)):
                for t in range(int(sc.decode_steps)):
                    if int(trt_tokens[i][t]) == int(vllm_tokens[i][t]):
                        match += 1
            vllm_acc = float(match) / float(max(1, total))

            vllm_perf = _bench_vllm_decode_tok_s(llm_eager_perf, prompt.tolist(), decode_steps=int(sc.decode_steps))
            vllm_tok_s = float(vllm_perf["decode_tok_s"])
            vllm_warm_s = float(vllm_perf["warm_s"])
            vllm_cached_s = float(vllm_perf["cached_s"])
            if llm_graph_perf is not None:
                vllm_perf_g = _bench_vllm_decode_tok_s(llm_graph_perf, prompt.tolist(), decode_steps=int(sc.decode_steps))
                vllm_tok_s_graph = float(vllm_perf_g["decode_tok_s"])
                vllm_warm_s_graph = float(vllm_perf_g["warm_s"])
                vllm_cached_s_graph = float(vllm_perf_g["cached_s"])

            prefix_lens = []
            for i in range(int(sc.bs)):
                p = 0
                for t in range(int(sc.decode_steps)):
                    if int(trt_tokens[i][t]) != int(vllm_tokens[i][t]):
                        break
                    p += 1
                prefix_lens.append(int(p))
            prefix_match_mean = float(sum(prefix_lens) / max(1, len(prefix_lens)))
            prefix_match_min = int(min(prefix_lens)) if prefix_lens else 0
            prefix_match_max = int(max(prefix_lens)) if prefix_lens else 0
            tf = {}
            if bool(args.teacher_force):
                tf = _trt_teacher_forced_next_token_match(
                    runtime, ctx, prompt, vllm_tokens, decode_steps=int(sc.decode_steps)
                )
        else:
            prefix_match_mean = ""
            prefix_match_min = ""
            prefix_match_max = ""
            tf = {}

        row = {
            "scenario": sc.name,
            "bs": int(sc.bs),
            "prefill_len": int(sc.prefill_len),
            "decode_steps": int(sc.decode_steps),
            "trt_decode_tok_s": float(trt_perf["decode_tok_s"]),
            "trt_decode_step_ms_mean": float(trt_perf["decode_step_ms_mean"]),
            "trt_decode_tok_s_cg": float(trt_perf_cg.get("decode_tok_s") or 0.0),
            "vllm_decode_tok_s": vllm_tok_s,
            "vllm_warm_s": vllm_warm_s,
            "vllm_cached_s": vllm_cached_s,
            "vllm_decode_tok_s_graph": vllm_tok_s_graph,
            "vllm_warm_s_graph": vllm_warm_s_graph,
            "vllm_cached_s_graph": vllm_cached_s_graph,
            "token_match_rate": vllm_acc,
            "prefix_match_mean": prefix_match_mean,
            "prefix_match_min": prefix_match_min,
            "prefix_match_max": prefix_match_max,
            **tf,
        }
        rows.append(row)
        print(row, flush=True)
        try:
            ctx.close()
        except Exception:
            pass
        del ctx
        torch.cuda.empty_cache()

    _ensure_dir(args.out_csv)
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["scenario"])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    md = _write_summary(rows, args.out_md)
    print({"summary": md}, flush=True)

    png = _maybe_plot(args.out_csv, args.out_png)
    if png:
        print({"plot": png}, flush=True)


if __name__ == "__main__":
    main()
