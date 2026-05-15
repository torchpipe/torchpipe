import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
import time
import zlib
from dataclasses import dataclass
from typing import Any
 
import torch
 
from orchid.llmscheduler.model_params import infer_model_params
from orchid.llmscheduler.runtime.trt_runtime import TensorRTModelRuntime
from orchid.paths import benchmark_artifact
from run_simple_suite import Scenario, _bench_trt_decode_tok_s, _make_ctx, _parse_scenarios, _preset_scenarios, _tokenizer_vocab
 
 
def _ensure_dir(path: str) -> None:
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)
 
 
def _prompt(seed: int, *, bs: int, prefill_len: int, vocab: int, scenario_name: str) -> torch.Tensor:
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed) ^ int(zlib.crc32(str(scenario_name).encode("utf-8")) & 0xFFFFFFFF))
    return torch.randint(low=0, high=int(vocab), size=(int(bs), int(prefill_len)), dtype=torch.int64, device="cpu", generator=g)
 
 
def _child_run_once() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--engine", default="")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--warmup-steps", type=int, default=10)
    ap.add_argument("--scenario-name", required=True)
    ap.add_argument("--bs", type=int, required=True)
    ap.add_argument("--prefill-len", type=int, required=True)
    ap.add_argument("--decode-steps", type=int, required=True)
    ap.add_argument("--freeze-lens", action="store_true")
    ap.add_argument("--trt-cudagraph", action="store_true")
    ap.add_argument("--cudagraph-mode", default="", choices=["", "off", "freeze", "piecewise"])
    ap.add_argument("--last-bucket", type=int, default=8)
    ap.add_argument("--attention-bypass", action="store_true")
    args = ap.parse_args()
 
    os.environ["LLMSCHEDULER_QUIET"] = "1"
    os.environ.setdefault("FLASHINFER_DISABLE_VERSION_CHECK", "1")
    os.environ.setdefault("LLMSCHEDULER_FLASHINFER_WORKSPACE_MB", "64")
    mode = str(args.cudagraph_mode or ("freeze" if bool(args.trt_cudagraph) else "off"))
    if mode == "off":
        os.environ["LLMSCHEDULER_TRT_CUDAGRAPH"] = "0"
    else:
        os.environ["LLMSCHEDULER_TRT_CUDAGRAPH"] = "1"
        os.environ["LLMSCHEDULER_TRT_CUDAGRAPH_FALLBACK"] = "1"
        if mode == "freeze":
            os.environ["LLMSCHEDULER_TRT_CUDAGRAPH_SINGLETON"] = "1"
            os.environ["LLMSCHEDULER_TRT_CUDAGRAPH_MAX_GRAPHS"] = "1"
            os.environ["LLMSCHEDULER_TRT_CUDAGRAPH_CTX_KEY"] = ""
        else:
            os.environ["LLMSCHEDULER_TRT_CUDAGRAPH_SINGLETON"] = "0"
            os.environ["LLMSCHEDULER_TRT_CUDAGRAPH_MAX_GRAPHS"] = "8"
            os.environ["LLMSCHEDULER_TRT_CUDAGRAPH_CTX_KEY"] = "page_and_last"
            os.environ["LLMSCHEDULER_TRT_CUDAGRAPH_LAST_BUCKET"] = str(int(args.last_bucket))
    os.environ["LLMSCHEDULER_ATTENTION_BYPASS"] = "1" if bool(args.attention_bypass) else "0"
 
    torch.manual_seed(int(args.seed))
    engine_path = str(args.engine).strip() or None
    vocab = _tokenizer_vocab(args.tokenizer)
 
    mp = infer_model_params(args.model, args.tokenizer)
    runtime = TensorRTModelRuntime(args.model, use_fp16=bool(args.fp16), engine_path=engine_path)
 
    sc = Scenario(str(args.scenario_name), int(args.bs), int(args.prefill_len), int(args.decode_steps))
    prompt = _prompt(int(args.seed), bs=int(sc.bs), prefill_len=int(sc.prefill_len), vocab=int(vocab), scenario_name=str(sc.name))
    ctx = _make_ctx(mp, bs=int(sc.bs), prefill_len=int(sc.prefill_len), decode_steps=int(sc.decode_steps), device=str(args.device), fp16=bool(args.fp16))
 
    if mode != "off":
        os.environ["LLMSCHEDULER_TRT_CUDAGRAPH"] = "0"
        _ = _bench_trt_decode_tok_s(
            runtime,
            ctx,
            prompt,
            decode_steps=int(sc.decode_steps),
            warmup_steps=max(1, int(args.warmup_steps)),
            vocab=int(vocab),
            freeze_lens=bool(args.freeze_lens),
        )
        torch.cuda.synchronize()
        os.environ["LLMSCHEDULER_TRT_CUDAGRAPH"] = "1"
 
    t0 = time.perf_counter()
    perf = _bench_trt_decode_tok_s(
        runtime,
        ctx,
        prompt,
        decode_steps=int(sc.decode_steps),
        warmup_steps=int(args.warmup_steps),
        vocab=int(vocab),
        freeze_lens=bool(args.freeze_lens),
    )
    torch.cuda.synchronize()
    t1 = time.perf_counter()
 
    try:
        ctx.close()
    except Exception:
        pass
    del ctx
    torch.cuda.empty_cache()
 
    out = {
        "scenario": str(sc.name),
        "bs": int(sc.bs),
        "prefill_len": int(sc.prefill_len),
        "decode_steps": int(sc.decode_steps),
        "freeze_lens": bool(args.freeze_lens),
        "trt_cudagraph": bool(mode != "off"),
        "trt_cudagraph_mode": str(mode),
        "last_bucket": int(args.last_bucket),
        "attention_bypass": bool(args.attention_bypass),
        "decode_tok_s": float(perf.get("decode_tok_s") or 0.0),
        "decode_step_ms_mean": float(perf.get("decode_step_ms_mean") or 0.0),
        "child_wall_s": float(t1 - t0),
    }
    sys.stdout.write(json.dumps(out, ensure_ascii=False))
 
 
def _run_child_once(
    *,
    model: str,
    tokenizer: str,
    engine: str,
    fp16: bool,
    device: str,
    seed: int,
    warmup_steps: int,
    scenario: Scenario,
    freeze_lens: bool,
    cudagraph_mode: str,
    last_bucket: int,
    attention_bypass: bool,
) -> dict[str, Any]:
    cmd = [
        sys.executable,
        os.path.abspath(__file__),
        "--child",
        "--model",
        str(model),
        "--tokenizer",
        str(tokenizer),
        "--engine",
        str(engine),
        "--device",
        str(device),
        "--seed",
        str(seed),
        "--warmup-steps",
        str(warmup_steps),
        "--scenario-name",
        str(scenario.name),
        "--bs",
        str(int(scenario.bs)),
        "--prefill-len",
        str(int(scenario.prefill_len)),
        "--decode-steps",
        str(int(scenario.decode_steps)),
    ]
    if bool(fp16):
        cmd.append("--fp16")
    if bool(freeze_lens):
        cmd.append("--freeze-lens")
    mode = str(cudagraph_mode or "off")
    if mode != "off":
        cmd.extend(["--cudagraph-mode", mode, "--last-bucket", str(int(last_bucket))])
    if bool(attention_bypass):
        cmd.append("--attention-bypass")
 
    env = dict(os.environ)
    env["LLMSCHEDULER_QUIET"] = "1"
    env.setdefault("LLMSCHEDULER_FLASHINFER_WORKSPACE_MB", "64")
    out = subprocess.check_output(cmd, env=env, stderr=subprocess.STDOUT, text=True)
    s = str(out).strip()
    i = s.rfind("{")
    if i >= 0:
        s = s[i:]
    return json.loads(s)
 
 
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
 
 
def _plot_bs_sweep(rows: list[dict[str, Any]], png_path: str, *, title: str) -> str | None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return None
 
    by_key: dict[tuple[bool, bool], list[dict[str, Any]]] = {}
    for r in rows:
        key = (bool(r["trt_cudagraph"]), bool(r["attention_bypass"]))
        by_key.setdefault(key, []).append(r)
 
    plt.figure(figsize=(8, 5))
    for (cg, bypass), rs in sorted(by_key.items(), key=lambda x: (x[0][0], x[0][1])):
        rs2 = sorted(rs, key=lambda x: int(x["bs"]))
        xs = [int(x["bs"]) for x in rs2]
        ys = [float(x["decode_tok_s"]) for x in rs2]
        label = f"{'TRT-cg' if cg else 'TRT'} / {'attn off' if bypass else 'attn on'}"
        plt.plot(xs, ys, marker="o", label=label)
 
    plt.xlabel("batch size")
    plt.ylabel("decode tok/s")
    plt.title(title)
    plt.grid(True, axis="y", alpha=0.2)
    plt.legend()
    _ensure_dir(png_path)
    plt.tight_layout()
    plt.savefig(png_path, dpi=150)
    plt.close()
    return png_path
 
 
def _write_summary_md(out_dir: str, *, ablation_rows: list[dict[str, Any]], sweep_rows: list[dict[str, Any]]) -> str:
    def key(r: dict[str, Any]) -> tuple[str, bool, bool, bool]:
        return (str(r["scenario"]), bool(r["freeze_lens"]), bool(r["trt_cudagraph"]), bool(r["attention_bypass"]))

    by = {key(r): r for r in ablation_rows}

    def get(scenario: str, *, freeze: bool, cg: bool, bypass: bool) -> float:
        r = by.get((str(scenario), bool(freeze), bool(cg), bool(bypass)))
        return float(r.get("decode_tok_s") or 0.0) if r else 0.0

    def fmt(x: float) -> str:
        return f"{x:.2f}"

    md_path = os.path.join(str(out_dir), "summary.md")
    _ensure_dir(md_path)
    with open(md_path, "w") as f:
        f.write("# Ablation summary\n\n")
        f.write("本目录的核心目标：判断 TRT CUDA graph 的收益主要来自 attention/flashinfer 还是来自 TRT 主干。\n\n")
        f.write("## 产物\n\n")
        f.write("- attention_bypass_ablation.csv\n")
        f.write("- fixed_decode_bs_sweep.csv\n")
        f.write("- fixed_decode_bs_sweep.png\n")
        f.write("- meta.json\n\n")

        scenarios = sorted({str(r["scenario"]) for r in ablation_rows})
        f.write("## 场景对照（freeze lens=true）\n\n")
        f.write("|scenario|TRT attn on|TRT attn off|TRT-cg attn on|TRT-cg attn off|cg speedup (attn on)|cg speedup (attn off)|\n")
        f.write("|-:|-:|-:|-:|-:|-:|-:|\n")
        for s in scenarios:
            trt_on = get(s, freeze=True, cg=False, bypass=False)
            trt_off = get(s, freeze=True, cg=False, bypass=True)
            cg_on = get(s, freeze=True, cg=True, bypass=False)
            cg_off = get(s, freeze=True, cg=True, bypass=True)
            sp_on = (cg_on / trt_on) if trt_on > 0 else 0.0
            sp_off = (cg_off / trt_off) if trt_off > 0 else 0.0
            f.write(f"|{s}|{fmt(trt_on)}|{fmt(trt_off)}|{fmt(cg_on)}|{fmt(cg_off)}|{sp_on:.3f}|{sp_off:.3f}|\n")

        f.write("\n")
        f.write("## Batch-size 曲线（freeze lens=true, prefill_len/decode_steps 固定）\n\n")

        def sweep_get(bs: int, *, cg: bool, bypass: bool) -> float:
            for r in sweep_rows:
                if int(r["bs"]) == int(bs) and bool(r["trt_cudagraph"]) == bool(cg) and bool(r["attention_bypass"]) == bool(bypass):
                    return float(r.get("decode_tok_s") or 0.0)
            return 0.0

        bs_list = sorted({int(r["bs"]) for r in sweep_rows})
        f.write("|bs|TRT on|TRT off|TRT-cg on|TRT-cg off|cg speedup on|cg speedup off|\n")
        f.write("|-:|-:|-:|-:|-:|-:|-:|\n")
        for bs in bs_list:
            trt_on = sweep_get(bs, cg=False, bypass=False)
            trt_off = sweep_get(bs, cg=False, bypass=True)
            cg_on = sweep_get(bs, cg=True, bypass=False)
            cg_off = sweep_get(bs, cg=True, bypass=True)
            sp_on = (cg_on / trt_on) if trt_on > 0 else 0.0
            sp_off = (cg_off / trt_off) if trt_off > 0 else 0.0
            f.write(f"|{bs}|{fmt(trt_on)}|{fmt(trt_off)}|{fmt(cg_on)}|{fmt(cg_off)}|{sp_on:.3f}|{sp_off:.3f}|\n")

        f.write("\n")
        f.write("## 读数建议\n\n")
        f.write("- 若 `cg speedup (attn on)` 明显大于 `cg speedup (attn off)`，说明 cudagraph 的主要收益来自 attention/flashinfer 路径（kernel launch/dispatch/元数据准备等）。\n")
        f.write("- 若两者接近，说明 cudagraph 的收益更多来自 TRT 主干（GEMM/LN 等）。\n")
    return md_path


def main() -> None:
    if "--child" in sys.argv:
        sys.argv = [x for x in sys.argv if x != "--child"]
        _child_run_once()
        return
 
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--engine", default="")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--warmup-steps", type=int, default=10)
    ap.add_argument("--preset", default="more")
    ap.add_argument("--scenarios", default="")
    ap.add_argument("--max-scenarios", type=int, default=3)
    ap.add_argument("--out-dir", default=benchmark_artifact("ablations"))
    ap.add_argument("--bs-sweep", default="1,2,4,8,16")
    ap.add_argument("--prefill-len", type=int, default=128)
    ap.add_argument("--decode-steps", type=int, default=128)
    args = ap.parse_args()
 
    out_dir = os.path.abspath(str(args.out_dir))
    os.makedirs(out_dir, exist_ok=True)
 
    os.environ["LLMSCHEDULER_QUIET"] = "1"
    os.environ.setdefault("FLASHINFER_DISABLE_VERSION_CHECK", "1")
    os.environ.setdefault("LLMSCHEDULER_FLASHINFER_WORKSPACE_MB", "64")
    scenarios = _parse_scenarios(str(args.scenarios)) or _preset_scenarios(str(args.preset))
    if int(args.max_scenarios) > 0:
        scenarios = list(scenarios[: int(args.max_scenarios)])
 
    mp = infer_model_params(args.model, args.tokenizer)
    runtime = TensorRTModelRuntime(args.model, use_fp16=bool(args.fp16), engine_path=(str(args.engine).strip() or None))
    vocab = _tokenizer_vocab(args.tokenizer)
 
    meta = {
        "model": str(args.model),
        "tokenizer": str(args.tokenizer),
        "engine": str(args.engine),
        "fp16": bool(args.fp16),
        "device": str(args.device),
        "seed": int(args.seed),
        "warmup_steps": int(args.warmup_steps),
        "preset": str(args.preset),
        "scenarios": [s.__dict__ for s in scenarios],
        "ts": int(time.time()),
    }
    meta_path = os.path.join(out_dir, "meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
 
    rows: list[dict[str, Any]] = []
    for sc in scenarios:
        for freeze_lens in (False, True):
            for bypass in (False, True):
                os.environ["LLMSCHEDULER_TRT_CUDAGRAPH"] = "0"
                os.environ["LLMSCHEDULER_ATTENTION_BYPASS"] = "1" if bool(bypass) else "0"
                prompt = _prompt(
                    int(args.seed),
                    bs=int(sc.bs),
                    prefill_len=int(sc.prefill_len),
                    vocab=int(vocab),
                    scenario_name=str(sc.name),
                )
                ctx = _make_ctx(
                    mp,
                    bs=int(sc.bs),
                    prefill_len=int(sc.prefill_len),
                    decode_steps=int(sc.decode_steps),
                    device=str(args.device),
                    fp16=bool(args.fp16),
                )
                perf = _bench_trt_decode_tok_s(
                    runtime,
                    ctx,
                    prompt,
                    decode_steps=int(sc.decode_steps),
                    warmup_steps=int(args.warmup_steps),
                    vocab=int(vocab),
                    freeze_lens=bool(freeze_lens),
                )
                row = {
                    "scenario": str(sc.name),
                    "bs": int(sc.bs),
                    "prefill_len": int(sc.prefill_len),
                    "decode_steps": int(sc.decode_steps),
                    "freeze_lens": bool(freeze_lens),
                    "trt_cudagraph": False,
                    "trt_cudagraph_mode": "off",
                    "last_bucket": 0,
                    "attention_bypass": bool(bypass),
                    "decode_tok_s": float(perf.get("decode_tok_s") or 0.0),
                    "decode_step_ms_mean": float(perf.get("decode_step_ms_mean") or 0.0),
                }
                rows.append(row)
                try:
                    ctx.close()
                except Exception:
                    pass
                del ctx
                torch.cuda.empty_cache()
 
        for bypass in (False, True):
            for mode in ("freeze", "piecewise"):
                r1 = _run_child_once(
                    model=str(args.model),
                    tokenizer=str(args.tokenizer),
                    engine=str(args.engine),
                    fp16=bool(args.fp16),
                    device=str(args.device),
                    seed=int(args.seed),
                    warmup_steps=int(args.warmup_steps),
                    scenario=sc,
                    freeze_lens=True,
                    cudagraph_mode=str(mode),
                    last_bucket=8,
                    attention_bypass=bool(bypass),
                )
                rows.append(r1)
 
    csv_path = _write_csv(rows, os.path.join(out_dir, "attention_bypass_ablation.csv"))
 
    bs_list = [int(x.strip()) for x in str(args.bs_sweep).split(",") if x.strip()]
    sweep_rows: list[dict[str, Any]] = []
    for bs in bs_list:
        sc = Scenario(f"bs{bs}_p{int(args.prefill_len)}_s{int(args.decode_steps)}", int(bs), int(args.prefill_len), int(args.decode_steps))
        for bypass in (False, True):
            for cg in ("off", "freeze", "piecewise"):
                if cg != "off":
                    r = _run_child_once(
                        model=str(args.model),
                        tokenizer=str(args.tokenizer),
                        engine=str(args.engine),
                        fp16=bool(args.fp16),
                        device=str(args.device),
                        seed=int(args.seed),
                        warmup_steps=int(args.warmup_steps),
                        scenario=sc,
                        freeze_lens=True,
                        cudagraph_mode=str(cg),
                        last_bucket=8,
                        attention_bypass=bool(bypass),
                    )
                else:
                    os.environ["LLMSCHEDULER_TRT_CUDAGRAPH"] = "0"
                    os.environ["LLMSCHEDULER_ATTENTION_BYPASS"] = "1" if bool(bypass) else "0"
                    prompt = _prompt(
                        int(args.seed),
                        bs=int(sc.bs),
                        prefill_len=int(sc.prefill_len),
                        vocab=int(vocab),
                        scenario_name=str(sc.name),
                    )
                    ctx = _make_ctx(
                        mp,
                        bs=int(sc.bs),
                        prefill_len=int(sc.prefill_len),
                        decode_steps=int(sc.decode_steps),
                        device=str(args.device),
                        fp16=bool(args.fp16),
                    )
                    perf = _bench_trt_decode_tok_s(
                        runtime,
                        ctx,
                        prompt,
                        decode_steps=int(sc.decode_steps),
                        warmup_steps=int(args.warmup_steps),
                        vocab=int(vocab),
                        freeze_lens=True,
                    )
                    r = {
                        "scenario": str(sc.name),
                        "bs": int(sc.bs),
                        "prefill_len": int(sc.prefill_len),
                        "decode_steps": int(sc.decode_steps),
                        "freeze_lens": True,
                        "trt_cudagraph": False,
                        "trt_cudagraph_mode": "off",
                        "last_bucket": 0,
                        "attention_bypass": bool(bypass),
                        "decode_tok_s": float(perf.get("decode_tok_s") or 0.0),
                        "decode_step_ms_mean": float(perf.get("decode_step_ms_mean") or 0.0),
                    }
                    try:
                        ctx.close()
                    except Exception:
                        pass
                    del ctx
                    torch.cuda.empty_cache()
                sweep_rows.append(r)
 
    sweep_csv = _write_csv(sweep_rows, os.path.join(out_dir, "fixed_decode_bs_sweep.csv"))
    sweep_png = _plot_bs_sweep(
        sweep_rows,
        os.path.join(out_dir, "fixed_decode_bs_sweep.png"),
        title="Fixed-shape decode (freeze lens): attention on/off curves",
    )
    summary_md = _write_summary_md(out_dir, ablation_rows=rows, sweep_rows=sweep_rows)
 
    summary = {
        "meta": meta_path,
        "attention_bypass_ablation_csv": csv_path,
        "fixed_decode_bs_sweep_csv": sweep_csv,
        "fixed_decode_bs_sweep_png": sweep_png or "",
        "summary_md": summary_md,
    }
    sys.stdout.write(json.dumps(summary, ensure_ascii=False))
 
 
if __name__ == "__main__":
    main()
