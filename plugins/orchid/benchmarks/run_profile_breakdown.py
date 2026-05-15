import argparse
import csv
import json
import os
import subprocess
import sys
import time
import zlib
from typing import Any


def _ensure_dir(path: str) -> None:
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)


def _write_csv(rows: list[dict[str, Any]], path: str) -> str:
    _ensure_dir(path)
    if not rows:
        with open(path, "w") as f:
            f.write("")
        return path
    fields: list[str] = []
    seen = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                fields.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return path


def _prompt(seed: int, *, bs: int, prefill_len: int, vocab: int, name: str):
    import torch

    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed) ^ int(zlib.crc32(str(name).encode("utf-8")) & 0xFFFFFFFF))
    return torch.randint(low=0, high=int(vocab), size=(int(bs), int(prefill_len)), dtype=torch.int64, device="cpu", generator=g)


def _child() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--engine", default="")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--warmup-steps", type=int, default=10)
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--bs", type=int, required=True)
    ap.add_argument("--prefill-len", type=int, required=True)
    ap.add_argument("--decode-steps", type=int, required=True)
    ap.add_argument("--attention-bypass", action="store_true")
    args = ap.parse_args()

    os.environ["LLMSCHEDULER_QUIET"] = "1"
    os.environ.setdefault("FLASHINFER_DISABLE_VERSION_CHECK", "1")
    os.environ["LLMSCHEDULER_PROFILE"] = "1"
    os.environ.setdefault("LLMSCHEDULER_FLASHINFER_WORKSPACE_MB", "64")
    os.environ["LLMSCHEDULER_ATTENTION_BYPASS"] = "1" if bool(args.attention_bypass) else "0"
    os.environ["LLMSCHEDULER_TRT_CUDAGRAPH"] = "0"

    import torch
    from orchid.llmscheduler.model_params import infer_model_params
    from orchid.llmscheduler.runtime.base import AttentionContext
    from orchid.llmscheduler.runtime.trt_runtime import TensorRTModelRuntime
    from orchid.paths import benchmark_artifact

    torch.manual_seed(int(args.seed))
    engine_path = str(args.engine).strip() or None

    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(str(args.tokenizer), trust_remote_code=True)
        vocab = int(getattr(tok, "vocab_size", 0) or 0)
        vocab = vocab if vocab > 0 else 151936
    except Exception:
        vocab = 151936

    mp = infer_model_params(str(args.model), str(args.tokenizer))
    page_size = int(mp.page_size)
    pages_per_req = int((int(args.prefill_len) + int(args.decode_steps) + int(page_size) - 1) // int(page_size))
    pages_per_layer_need = int(int(args.bs) * int(max(1, pages_per_req)))
    max_pages = int(max(int(mp.max_pages), int(pages_per_layer_need) * int(mp.num_layers)))
    max_pages = max(max_pages, 20000)
    ctx = AttentionContext(
        num_layers=int(mp.num_layers),
        num_heads=int(mp.num_heads),
        kv_num_heads=int(mp.kv_num_heads),
        head_dim=int(mp.head_dim),
        page_size=int(page_size),
        max_pages=int(max_pages),
        use_cpp_metadata=True,
        device=str(args.device),
        use_fp16=bool(args.fp16),
    )

    runtime = TensorRTModelRuntime(str(args.model), use_fp16=bool(args.fp16), engine_path=engine_path)
    prompt = _prompt(int(args.seed), bs=int(args.bs), prefill_len=int(args.prefill_len), vocab=int(vocab), name=str(args.scenario))

    bs = int(args.bs)
    prefill_len = int(args.prefill_len)
    decode_steps = int(args.decode_steps)

    ctx._prof = None
    ctx._prof_events = None
    input_ids = prompt.reshape(-1).contiguous().to(ctx.device, dtype=torch.int32)
    ctx.current_batch_req_ids = list(range(int(bs)))
    ctx.current_batch_seq_lens = [int(prefill_len)] * int(bs)
    ctx.current_batch_total_lens = [int(prefill_len)] * int(bs)
    ctx.current_batch_history_lens = [0] * int(bs)
    ctx.current_batch_is_prefill = [True] * int(bs)
    ctx.is_all_decode = False
    _ = runtime.forward(input_ids, ctx)
    _ = runtime.forward(input_ids, ctx)

    ctx._prof = {}
    ctx._prof_events = {}

    tok = torch.randint(low=0, high=int(vocab), size=(int(bs),), dtype=torch.int32, device=ctx.device)
    history = [int(prefill_len)] * int(bs)
    total = [int(prefill_len) + 1] * int(bs)
    for i in range(int(args.warmup_steps)):
        ctx._engine_step_id = int(i)
        ctx.current_batch_req_ids = list(range(int(bs)))
        ctx.current_batch_seq_lens = [1] * int(bs)
        ctx.current_batch_history_lens = list(history)
        ctx.current_batch_total_lens = list(total)
        ctx.current_batch_is_prefill = [False] * int(bs)
        ctx.is_all_decode = True
        _ = runtime.forward(tok, ctx)
        history = [int(h) + 1 for h in history]
        total = [int(t) + 1 for t in total]

    t0 = time.perf_counter()
    for i in range(int(decode_steps)):
        ctx._engine_step_id = int(i) + int(args.warmup_steps)
        ctx.current_batch_req_ids = list(range(int(bs)))
        ctx.current_batch_seq_lens = [1] * int(bs)
        ctx.current_batch_history_lens = list(history)
        ctx.current_batch_total_lens = list(total)
        ctx.current_batch_is_prefill = [False] * int(bs)
        ctx.is_all_decode = True
        _ = runtime.forward(tok, ctx)
        history = [int(h) + 1 for h in history]
        total = [int(t) + 1 for t in total]
    torch.cuda.synchronize()
    wall_ms = float((time.perf_counter() - t0) * 1000.0)

    events_ms: dict[str, float] = {}
    for k, pairs in (ctx._prof_events or {}).items():
        s = 0.0
        for e0, e1 in pairs:
            try:
                s += float(e0.elapsed_time(e1))
            except Exception:
                s += 0.0
        events_ms[str(k)] = float(s)

    prof = dict(ctx._prof or {})
    out = {
        "scenario": str(args.scenario),
        "bs": int(bs),
        "prefill_len": int(prefill_len),
        "decode_steps": int(decode_steps),
        "attention_bypass": bool(args.attention_bypass),
        "wall_ms": float(wall_ms),
        "meta_ms": float(prof.get("meta_ms") or 0.0),
        "kv_write_ms": float(prof.get("kv_write_ms") or 0.0),
        "flashinfer_decode_ms": float(prof.get("flashinfer_decode_ms") or 0.0),
        "flashinfer_prefill_ms": float(prof.get("flashinfer_prefill_ms") or 0.0),
        "attn_impl_ms": float(prof.get("attn_impl_ms") or 0.0),
        "kv_write_gpu_ms": float(events_ms.get("kv_write") or 0.0),
        "flashinfer_gpu_ms": float(events_ms.get("flashinfer") or 0.0),
        "prof": prof,
        "events_ms": events_ms,
    }

    try:
        ctx.close()
    except Exception:
        pass
    del ctx
    torch.cuda.empty_cache()

    sys.stdout.write("LLMSCHEDULER_JSON=" + json.dumps(out, ensure_ascii=False))


def _run_child(args, *, scenario: dict[str, Any], attention_bypass: bool) -> dict[str, Any]:
    cmd = [
        sys.executable,
        os.path.abspath(__file__),
        "--child",
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
        "--scenario",
        str(scenario["name"]),
        "--bs",
        str(int(scenario["bs"])),
        "--prefill-len",
        str(int(scenario["prefill_len"])),
        "--decode-steps",
        str(int(scenario["decode_steps"])),
    ]
    if bool(args.fp16):
        cmd.append("--fp16")
    if bool(attention_bypass):
        cmd.append("--attention-bypass")
    env = dict(os.environ)
    env["LLMSCHEDULER_QUIET"] = "1"
    env.setdefault("FLASHINFER_DISABLE_VERSION_CHECK", "1")
    env.setdefault("LLMSCHEDULER_FLASHINFER_WORKSPACE_MB", "64")
    out = subprocess.check_output(cmd, env=env, stderr=subprocess.STDOUT, text=True)
    s = str(out)
    marker = "LLMSCHEDULER_JSON="
    j = s.rfind(marker)
    if j < 0:
        raise RuntimeError("Missing LLMSCHEDULER_JSON output from child")
    payload = s[j + len(marker) :].strip()
    dec = json.JSONDecoder()
    obj, _ = dec.raw_decode(payload)
    return obj


def main() -> None:
    if "--child" in sys.argv:
        sys.argv = [x for x in sys.argv if x != "--child"]
        _child()
        return

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--engine", default="")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--warmup-steps", type=int, default=10)
    ap.add_argument("--out-dir", default=benchmark_artifact("profile_breakdown"))
    ap.add_argument("--scenarios", default="bs1_p128_s64,1,128,64;bs2_p128_s128,2,128,128;bs10_p128_s128,10,128,128")
    args = ap.parse_args()

    out_dir = os.path.abspath(str(args.out_dir))
    os.makedirs(out_dir, exist_ok=True)

    scenarios: list[dict[str, Any]] = []
    for chunk in str(args.scenarios).split(";"):
        c = chunk.strip()
        if not c:
            continue
        parts = [x.strip() for x in c.split(",")]
        if len(parts) != 4:
            raise ValueError(f"Bad scenario spec: {c}")
        name, bs, p, steps = parts
        scenarios.append({"name": str(name), "bs": int(bs), "prefill_len": int(p), "decode_steps": int(steps)})

    meta = {
        "model": str(args.model),
        "tokenizer": str(args.tokenizer),
        "engine": str(args.engine),
        "fp16": bool(args.fp16),
        "device": str(args.device),
        "seed": int(args.seed),
        "warmup_steps": int(args.warmup_steps),
        "scenarios": scenarios,
        "ts": int(time.time()),
    }
    meta_path = os.path.join(out_dir, "meta.json")
    _ensure_dir(meta_path)
    with open(meta_path, "w") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    rows: list[dict[str, Any]] = []
    for sc in scenarios:
        for bypass in (False, True):
            r = _run_child(args, scenario=sc, attention_bypass=bool(bypass))
            prof = dict(r.get("prof") or {})
            events = dict(r.get("events_ms") or {})
            row = {
                "scenario": str(r.get("scenario")),
                "bs": int(r.get("bs") or 0),
                "prefill_len": int(r.get("prefill_len") or 0),
                "decode_steps": int(r.get("decode_steps") or 0),
                "attention_bypass": bool(r.get("attention_bypass")),
                "wall_ms": float(r.get("wall_ms") or 0.0),
                "meta_ms": float(r.get("meta_ms") or 0.0),
                "kv_write_ms": float(r.get("kv_write_ms") or 0.0),
                "flashinfer_decode_ms": float(r.get("flashinfer_decode_ms") or 0.0),
                "kv_write_gpu_ms": float(r.get("kv_write_gpu_ms") or 0.0),
                "flashinfer_gpu_ms": float(r.get("flashinfer_gpu_ms") or 0.0),
                "attn_impl_ms": float(r.get("attn_impl_ms") or 0.0),
                "batch_ms": float(prof.get("batch_ms") or 0.0),
                "trt_ms": float(prof.get("trt_ms") or 0.0),
                "sample_ms": float(prof.get("sample_ms") or 0.0),
            }
            rows.append(row)

    csv_path = _write_csv(rows, os.path.join(out_dir, "profile_breakdown.csv"))

    md_path = os.path.join(out_dir, "summary.md")
    _ensure_dir(md_path)
    with open(md_path, "w") as f:
        f.write("# Profile breakdown summary\n\n")
        f.write("- 读数：meta_ms/kv_write_ms/flashinfer_decode_ms 为 Python 侧 wall time 累计；kv_write_gpu_ms/flashinfer_gpu_ms 为 GPU event 累计。\n")
        f.write("- 对照：attention_bypass=false vs true，可近似隔离 attention/flashinfer 路径的开销与 cudagraph 潜在收益来源。\n\n")
        f.write(f"- meta: {os.path.relpath(meta_path, os.getcwd())}\n")
        f.write(f"- csv: {os.path.relpath(csv_path, os.getcwd())}\n\n")
        f.write("|scenario|bs|attn|wall ms|meta ms|kv write ms|flashinfer decode ms|kv gpu ms|flashinfer gpu ms|\n")
        f.write("|-:|-:|-:|-:|-:|-:|-:|-:|-:|\n")
        for r in rows:
            f.write(
                f"|{r['scenario']}|{r['bs']}|{'off' if r['attention_bypass'] else 'on'}|{r['wall_ms']:.2f}|{r['meta_ms']:.2f}|{r['kv_write_ms']:.2f}|{r['flashinfer_decode_ms']:.2f}|{r['kv_write_gpu_ms']:.2f}|{r['flashinfer_gpu_ms']:.2f}|\n"
            )

    sys.stdout.write(json.dumps({"out_dir": out_dir, "meta": meta_path, "csv": csv_path, "md": md_path}, ensure_ascii=False))


if __name__ == "__main__":
    main()
