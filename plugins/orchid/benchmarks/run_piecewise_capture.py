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
    ap.add_argument("--scenario", default="bs10_p120_s16")
    ap.add_argument("--bs", type=int, default=10)
    ap.add_argument("--prefill-len", type=int, default=120)
    ap.add_argument("--decode-steps", type=int, default=16)
    ap.add_argument("--mode", choices=["baseline", "freeze", "piecewise"], required=True)
    ap.add_argument("--attention-bypass", action="store_true")
    ap.add_argument("--last-bucket", type=int, default=8)
    args = ap.parse_args()

    os.environ["LLMSCHEDULER_QUIET"] = "1"
    os.environ.setdefault("FLASHINFER_DISABLE_VERSION_CHECK", "1")
    os.environ.setdefault("LLMSCHEDULER_FLASHINFER_WORKSPACE_MB", "64")
    os.environ["LLMSCHEDULER_ATTENTION_BYPASS"] = "1" if bool(args.attention_bypass) else "0"

    if str(args.mode) == "baseline":
        os.environ["LLMSCHEDULER_TRT_CUDAGRAPH"] = "0"
    elif str(args.mode) == "freeze":
        os.environ["LLMSCHEDULER_TRT_CUDAGRAPH"] = "1"
        os.environ["LLMSCHEDULER_TRT_CUDAGRAPH_FALLBACK"] = "1"
        os.environ["LLMSCHEDULER_TRT_CUDAGRAPH_SINGLETON"] = "1"
        os.environ["LLMSCHEDULER_TRT_CUDAGRAPH_MAX_GRAPHS"] = "1"
        os.environ["LLMSCHEDULER_TRT_CUDAGRAPH_CTX_KEY"] = ""
    else:
        os.environ["LLMSCHEDULER_TRT_CUDAGRAPH"] = "1"
        os.environ["LLMSCHEDULER_TRT_CUDAGRAPH_FALLBACK"] = "1"
        os.environ["LLMSCHEDULER_TRT_CUDAGRAPH_SINGLETON"] = "0"
        os.environ["LLMSCHEDULER_TRT_CUDAGRAPH_MAX_GRAPHS"] = "8"
        os.environ["LLMSCHEDULER_TRT_CUDAGRAPH_CTX_KEY"] = "page_and_last"
        os.environ["LLMSCHEDULER_TRT_CUDAGRAPH_LAST_BUCKET"] = str(int(args.last_bucket))

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

    os.environ["LLMSCHEDULER_TRT_CUDAGRAPH"] = "0"
    input_ids = prompt.reshape(-1).contiguous().to(ctx.device, dtype=torch.int32)
    ctx.current_batch_req_ids = list(range(int(bs)))
    ctx.current_batch_seq_lens = [int(prefill_len)] * int(bs)
    ctx.current_batch_total_lens = [int(prefill_len)] * int(bs)
    ctx.current_batch_history_lens = [0] * int(bs)
    ctx.current_batch_is_prefill = [True] * int(bs)
    ctx.is_all_decode = False
    _ = runtime.forward(input_ids, ctx)
    _ = runtime.forward(input_ids, ctx)

    if str(args.mode) != "baseline":
        os.environ["LLMSCHEDULER_TRT_CUDAGRAPH"] = "1"

    tok = torch.randint(low=0, high=int(vocab), size=(int(bs),), dtype=torch.int32, device=ctx.device)

    def run_pass(*, freeze_lens: bool, capture_only: bool) -> float:
        history = [int(prefill_len)] * int(bs)
        total = [int(prefill_len) + 1] * int(bs)
        t0 = time.perf_counter()
        for i in range(int(decode_steps)):
            ctx._engine_step_id = int(i)
            ctx.current_batch_req_ids = list(range(int(bs)))
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
        if bool(capture_only):
            return 0.0
        dt = float(time.perf_counter() - t0)
        return (float(bs) * float(decode_steps)) / float(dt) if dt > 0 else 0.0

    if str(args.mode) == "baseline":
        for _ in range(int(args.warmup_steps)):
            _ = run_pass(freeze_lens=False, capture_only=True)
        tok_s = run_pass(freeze_lens=False, capture_only=False)
    elif str(args.mode) == "freeze":
        _ = run_pass(freeze_lens=True, capture_only=True)
        tok_s = run_pass(freeze_lens=True, capture_only=False)
    else:
        _ = run_pass(freeze_lens=False, capture_only=True)
        tok_s = run_pass(freeze_lens=False, capture_only=False)

    graphs = 0
    try:
        graphs = int(len(getattr(runtime.runtime, "_cudagraph_cache", {}) or {}))
    except Exception:
        graphs = 0

    out = {
        "mode": str(args.mode),
        "scenario": str(args.scenario),
        "bs": int(bs),
        "prefill_len": int(prefill_len),
        "decode_steps": int(decode_steps),
        "page_size": int(page_size),
        "attention_bypass": bool(args.attention_bypass),
        "tok_s": float(tok_s),
        "graphs": int(graphs),
        "status": "ok",
    }

    try:
        ctx.close()
    except Exception:
        pass
    del ctx
    torch.cuda.empty_cache()

    sys.stdout.write("LLMSCHEDULER_JSON=" + json.dumps(out, ensure_ascii=False))


def _run_child(args, *, mode: str) -> dict[str, Any]:
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
        str(args.scenario),
        "--bs",
        str(int(args.bs)),
        "--prefill-len",
        str(int(args.prefill_len)),
        "--decode-steps",
        str(int(args.decode_steps)),
        "--mode",
        str(mode),
    ]
    if bool(args.fp16):
        cmd.append("--fp16")
    if bool(getattr(args, "attention_bypass", False)):
        cmd.append("--attention-bypass")
    if int(getattr(args, "last_bucket", 0) or 0) > 0:
        cmd.extend(["--last-bucket", str(int(getattr(args, "last_bucket")))])
    env = dict(os.environ)
    env["LLMSCHEDULER_QUIET"] = "1"
    env.setdefault("LLMSCHEDULER_FLASHINFER_WORKSPACE_MB", "64")
    try:
        out = subprocess.check_output(cmd, env=env, stderr=subprocess.STDOUT, text=True)
    except subprocess.CalledProcessError as e:
        return {
            "mode": str(mode),
            "scenario": str(args.scenario),
            "bs": int(args.bs),
            "prefill_len": int(args.prefill_len),
            "decode_steps": int(args.decode_steps),
            "attention_bypass": bool(getattr(args, "attention_bypass", False)),
            "tok_s": 0.0,
            "graphs": 0,
            "status": "failed",
            "error": str(e),
        }
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
    ap.add_argument("--warmup-steps", type=int, default=3)
    ap.add_argument("--scenario", default="bs10_p120_s16")
    ap.add_argument("--bs", type=int, default=10)
    ap.add_argument("--prefill-len", type=int, default=120)
    ap.add_argument("--decode-steps", type=int, default=16)
    ap.add_argument("--attention-bypass", action="store_true")
    ap.add_argument("--last-bucket", type=int, default=8)
    ap.add_argument("--out-dir", default=benchmark_artifact("piecewise_capture"))
    args = ap.parse_args()

    out_dir = os.path.abspath(str(args.out_dir))
    os.makedirs(out_dir, exist_ok=True)

    meta = {
        "model": str(args.model),
        "tokenizer": str(args.tokenizer),
        "engine": str(args.engine),
        "fp16": bool(args.fp16),
        "device": str(args.device),
        "seed": int(args.seed),
        "warmup_steps": int(args.warmup_steps),
        "scenario": str(args.scenario),
        "bs": int(args.bs),
        "prefill_len": int(args.prefill_len),
        "decode_steps": int(args.decode_steps),
        "ts": int(time.time()),
    }
    meta_path = os.path.join(out_dir, "meta.json")
    _ensure_dir(meta_path)
    with open(meta_path, "w") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    rows = []
    for mode in ("baseline", "freeze", "piecewise"):
        rows.append(_run_child(args, mode=str(mode)))
    csv_path = _write_csv(rows, os.path.join(out_dir, "piecewise_capture.csv"))

    png_path = None
    try:
        import matplotlib.pyplot as plt

        names = [r["mode"] for r in rows]
        vals = [float(r["tok_s"]) for r in rows]
        plt.figure(figsize=(7, 4))
        plt.bar(names, vals)
        plt.ylabel("decode tok/s")
        plt.title("TRT cudagraph: baseline vs freeze vs piecewise")
        plt.grid(True, axis="y", alpha=0.2)
        png_path = os.path.join(out_dir, "piecewise_capture.png")
        plt.tight_layout()
        plt.savefig(png_path, dpi=150)
        plt.close()
    except Exception:
        png_path = None

    md_path = os.path.join(out_dir, "summary.md")
    _ensure_dir(md_path)
    by = {r["mode"]: r for r in rows}
    with open(md_path, "w") as f:
        f.write("# Piecewise capture summary\n\n")
        f.write(f"- meta: {os.path.relpath(meta_path, os.getcwd())}\n")
        f.write(f"- csv: {os.path.relpath(csv_path, os.getcwd())}\n")
        if png_path:
            f.write(f"- png: {os.path.relpath(png_path, os.getcwd())}\n")
        f.write("\n")
        f.write("|mode|tok/s|graphs|\n")
        f.write("|-:|-:|-:|\n")
        for r in rows:
            f.write(f"|{r['mode']}|{float(r['tok_s']):.2f}|{int(r.get('graphs') or 0)}|\n")
        f.write("\n")
        if "baseline" in by and "piecewise" in by and float(by["baseline"]["tok_s"]) > 0:
            f.write(f"- piecewise speedup over baseline: {float(by['piecewise']['tok_s'])/float(by['baseline']['tok_s']):.4f}\n")
        if "baseline" in by and "freeze" in by and float(by["baseline"]["tok_s"]) > 0:
            f.write(f"- freeze speedup over baseline: {float(by['freeze']['tok_s'])/float(by['baseline']['tok_s']):.4f}\n")

    sys.stdout.write(json.dumps({"out_dir": out_dir, "meta": meta_path, "csv": csv_path, "png": png_path or "", "md": md_path}, ensure_ascii=False))


if __name__ == "__main__":
    main()
