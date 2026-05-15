from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import urlopen


@dataclass
class Scenario:
    name: str
    dataset: str
    parallel: list[int]
    number: list[int]
    max_tokens: int
    min_tokens: int | None = None
    prefix_length: int | None = None
    min_prompt_length: int | None = None
    max_prompt_length: int | None = None
    stream: bool = False


def _benchmarks_root() -> Path:
    return Path(__file__).resolve().parent


def _default_out_dir() -> Path:
    return _benchmarks_root() / "artifacts" / "evalscope_perf"


def _scenarios_for_suite(suite: str) -> list[Scenario]:
    suites: dict[str, list[Scenario]] = {
        "smoke": [
            Scenario("random_short", "random", [1, 2], [6, 6], 32, 32, 0, 128, 128),
            Scenario("random_medium", "random", [1, 2], [6, 6], 64, 64, 0, 256, 256),
        ],
        "standard": [
            Scenario("random_short", "random", [1, 4, 8], [12, 12, 12], 64, 64, 0, 256, 256),
            Scenario("random_long", "random", [1, 4, 8], [10, 10, 10], 128, 128, 0, 1024, 1024),
            Scenario("openqa_stream", "openqa", [1, 4], [16, 16], 128, stream=True),
        ],
        "full": [
            Scenario("random_short", "random", [1, 4, 8, 16], [20, 20, 20, 20], 64, 64, 0, 256, 256),
            Scenario("random_long", "random", [1, 4, 8, 10], [20, 20, 20, 20], 128, 128, 0, 1024, 1024),
            Scenario("openqa_stream", "openqa", [1, 4, 8], [24, 24, 24], 128, stream=True),
        ],
    }
    return suites[suite]


def _ensure_binary(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise SystemExit(f"missing required executable: {name}")
    return path


def _wait_for_http(url: str, timeout_s: float) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with urlopen(url, timeout=2.0) as resp:
                if 200 <= int(resp.status) < 500:
                    return
        except URLError:
            pass
        except Exception:
            pass
        time.sleep(1.0)
    raise TimeoutError(f"service not ready: {url}")


def _find_latest_output(run_dir: Path) -> Path | None:
    outputs_root = run_dir / "outputs"
    if not outputs_root.exists():
        return None
    candidates = [p for p in outputs_root.rglob("*") if p.is_dir()]
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    for candidate in candidates:
        if (candidate / "performance_summary.txt").exists() or (candidate / "benchmark.log").exists():
            return candidate
    return candidates[0]


def _parse_summary_table(summary_path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    if not summary_path.exists():
        return result
    rows: list[list[str]] = []
    for line in summary_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line.startswith("|") or line.count("|") < 3:
            if line.startswith("│") and line.count("│") >= 10:
                parts = [part.strip() for part in line.strip("│").split("│")]
                if parts and parts[0].isdigit():
                    rows.append(parts)
            continue
        parts = [part.strip() for part in line.strip("|").split("|")]
        if len(parts) == 2:
            key, value = parts
            if key not in {"Key", ""} and not set(key) <= {"-", "="}:
                result[key] = value
    if rows:
        row = rows[-1]
        if len(row) >= 10:
            result["Request throughput (req/s)"] = row[2]
            result["Average time to first token (s)"] = row[5]
            result["Average time per output token (s)"] = row[7]
            result["Output token throughput (tok/s)"] = row[9]
    return result


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(str(value).strip())
    except Exception:
        return None


def _make_evalscope_cmd(
    scenario: Scenario,
    url: str,
    model: str,
    tokenizer_path: str,
    extra_args: dict[str, Any],
) -> list[str]:
    cmd = [
        _ensure_binary("evalscope"),
        "perf",
        "--parallel",
        *[str(x) for x in scenario.parallel],
        "--number",
        *[str(x) for x in scenario.number],
        "--model",
        model,
        "--url",
        url,
        "--api",
        "openai",
        "--dataset",
        scenario.dataset,
        "--max-tokens",
        str(scenario.max_tokens),
        "--extra-args",
        json.dumps(extra_args, ensure_ascii=False),
    ]
    if scenario.dataset == "random":
        cmd.extend(
            [
                "--min-tokens",
                str(scenario.min_tokens if scenario.min_tokens is not None else scenario.max_tokens),
                "--prefix-length",
                str(scenario.prefix_length if scenario.prefix_length is not None else 0),
                "--min-prompt-length",
                str(scenario.min_prompt_length if scenario.min_prompt_length is not None else scenario.max_tokens),
                "--max-prompt-length",
                str(scenario.max_prompt_length if scenario.max_prompt_length is not None else scenario.max_tokens),
                "--tokenizer-path",
                tokenizer_path,
            ]
        )
    if scenario.stream:
        cmd.append("--stream")
    return cmd


def _spawn_vllm(args: argparse.Namespace, name: str) -> tuple[subprocess.Popen[str], str]:
    python_bin = sys.executable
    port = int(args.vllm_port)
    cmd = [
        python_bin,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        args.vllm_model or args.model,
        "--tokenizer",
        args.tokenizer_path,
        "--served-model-name",
        args.model,
        "--host",
        args.host,
        "--port",
        str(port),
        "--dtype",
        "float16",
        "--tensor-parallel-size",
        "1",
        "--max-model-len",
        str(args.vllm_max_model_len),
        "--gpu-memory-utilization",
        str(args.vllm_gpu_mem),
        "--no-enable-log-requests",
    ]
    if args.vllm_enforce_eager:
        cmd.append("--enforce-eager")
    log_path = Path(args.out_dir) / name / "vllm_server.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_fp = log_path.open("w", encoding="utf-8")
    proc = subprocess.Popen(cmd, stdout=log_fp, stderr=subprocess.STDOUT, text=True)
    url = f"http://{args.host}:{port}/v1/chat/completions"
    _wait_for_http(f"http://{args.host}:{port}/v1/models", timeout_s=args.server_timeout_s)
    return proc, url


def _spawn_orchid(args: argparse.Namespace, name: str) -> tuple[subprocess.Popen[str], str]:
    python_bin = sys.executable
    port = int(args.orchid_port)
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{args.orchid_src}:{env.get('PYTHONPATH', '')}" if env.get("PYTHONPATH") else str(args.orchid_src)
    env["MODEL_PATH"] = args.orchid_model_path
    env["TOKENIZER_PATH"] = args.tokenizer_path
    env["ENGINE_PATH"] = args.orchid_engine_path
    env["LLMSCHEDULER_HOST"] = args.host
    env["LLMSCHEDULER_PORT"] = str(port)
    env["LLMSCHEDULER_USE_FP16"] = "1"
    env["LLMSCHEDULER_PAGE_SIZE"] = str(args.orchid_page_size)
    env["LLMSCHEDULER_MAX_PAGES"] = str(args.orchid_max_pages)
    env["LLMSCHEDULER_GPU_MEMORY_UTILIZATION"] = str(args.orchid_gpu_mem)
    env["LLMSCHEDULER_KV_CACHE_RESERVED_MB"] = str(args.orchid_kv_cache_reserved_mb)
    env["LLMSCHEDULER_APPLY_CHAT_TEMPLATE"] = "1"
    env["LLMSCHEDULER_LOG_STARTUP"] = "1"
    env["FLASHINFER_DISABLE_VERSION_CHECK"] = env.get("FLASHINFER_DISABLE_VERSION_CHECK", "1")
    env["LLMSCHEDULER_FLASHINFER_DECODE_CUDAGRAPH"] = env.get("LLMSCHEDULER_FLASHINFER_DECODE_CUDAGRAPH", "0")
    env["LLMSCHEDULER_FLASHINFER_DISABLE_SPLIT_KV"] = env.get("LLMSCHEDULER_FLASHINFER_DISABLE_SPLIT_KV", "1")
    env["LLMSCHEDULER_FLASHINFER_WORKSPACE_MB"] = env.get("LLMSCHEDULER_FLASHINFER_WORKSPACE_MB", "512")
    env["LLMSCHEDULER_MAX_BATCH_TOKENS"] = env.get("LLMSCHEDULER_MAX_BATCH_TOKENS", "4096")
    env["LLMSCHEDULER_TRT_CUDAGRAPH"] = env.get("LLMSCHEDULER_TRT_CUDAGRAPH", "1")
    env["LLMSCHEDULER_TRT_CUDAGRAPH_CTX_KEY"] = env.get("LLMSCHEDULER_TRT_CUDAGRAPH_CTX_KEY", "page_and_last")
    env["LLMSCHEDULER_TRT_CUDAGRAPH_LAST_BUCKET"] = env.get("LLMSCHEDULER_TRT_CUDAGRAPH_LAST_BUCKET", "8")
    env["LLMSCHEDULER_TRT_CUDAGRAPH_MAX_GRAPHS"] = env.get("LLMSCHEDULER_TRT_CUDAGRAPH_MAX_GRAPHS", "8")
    env["LLMSCHEDULER_TRT_CUDAGRAPH_FALLBACK"] = env.get("LLMSCHEDULER_TRT_CUDAGRAPH_FALLBACK", "1")
    env["LLMSCHEDULER_TRT_USE_TORCH_STREAM"] = env.get("LLMSCHEDULER_TRT_USE_TORCH_STREAM", "0")
    env["LLMSCHEDULER_TRT_INPUT_IDS_PROFILES"] = env.get("LLMSCHEDULER_TRT_INPUT_IDS_PROFILES", "1,32,64;64,512,1024;1024,3072,4096")
    env["LLMSCHEDULER_TRT_TIMING_CACHE"] = env.get("LLMSCHEDULER_TRT_TIMING_CACHE", str(Path.home() / ".cache" / "orchid" / "trt_timing_cache.bin"))
    cmd = [python_bin, "-m", "orchid.llmscheduler.server.api_server"]
    log_path = Path(args.out_dir) / name / "orchid_server.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_fp = log_path.open("w", encoding="utf-8")
    proc = subprocess.Popen(cmd, stdout=log_fp, stderr=subprocess.STDOUT, text=True, env=env)
    url = f"http://{args.host}:{port}/v1/chat/completions"
    _wait_for_http(f"http://{args.host}:{port}/health", timeout_s=args.server_timeout_s)
    return proc, url


def _run_target_scenario(
    target_name: str,
    url: str,
    scenario: Scenario,
    args: argparse.Namespace,
) -> dict[str, Any]:
    run_dir = Path(args.out_dir) / target_name / scenario.name
    run_dir.mkdir(parents=True, exist_ok=True)
    cmd = _make_evalscope_cmd(
        scenario=scenario,
        url=url,
        model=args.model,
        tokenizer_path=args.tokenizer_path,
        extra_args={"ignore_eos": True, **args.extra_args},
    )
    proc = subprocess.run(cmd, cwd=run_dir, env=os.environ.copy(), capture_output=True, text=True)
    (run_dir / "stdout.log").write_text(proc.stdout, encoding="utf-8")
    (run_dir / "stderr.log").write_text(proc.stderr, encoding="utf-8")
    latest_output = _find_latest_output(run_dir)
    summary_metrics: dict[str, str] = {}
    summary_path: Path | None = None
    if latest_output is not None:
        summary_path = latest_output / "performance_summary.txt"
        summary_metrics = _parse_summary_table(summary_path)
    return {
        "target": target_name,
        "url": url,
        "scenario": asdict(scenario),
        "command": cmd,
        "returncode": proc.returncode,
        "run_dir": str(run_dir),
        "output_dir": str(latest_output) if latest_output is not None else None,
        "performance_summary": str(summary_path) if summary_path is not None and summary_path.exists() else None,
        "summary_metrics": summary_metrics,
    }


def _write_report(results: list[dict[str, Any]], args: argparse.Namespace) -> Path:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "meta.json").write_text(json.dumps({"args": vars(args), "results": results}, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# EvalScope Perf Summary",
        "",
        f"- model: `{args.model}`",
        f"- tokenizer: `{args.tokenizer_path}`",
        f"- suite: `{args.suite}`",
        "",
        "| target | scenario | rc | req/s | out tok/s | avg ttft(s) | avg tpot(s) | output dir |",
        "| - | - | -: | -: | -: | -: | -: | - |",
    ]
    for item in results:
        metrics = item.get("summary_metrics") or {}
        lines.append(
            "| "
            + " | ".join(
                [
                    str(item["target"]),
                    str(item["scenario"]["name"]),
                    str(item["returncode"]),
                    str(metrics.get("Request throughput (req/s)", "")),
                    str(metrics.get("Output token throughput (tok/s)", "")),
                    str(metrics.get("Average time to first token (s)", "")),
                    str(metrics.get("Average time per output token (s)", "")),
                    str(item.get("output_dir") or ""),
                ]
            )
            + " |"
        )
    report_path = out_dir / "summary.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def _pick_compare_targets(results: list[dict[str, Any]]) -> list[str]:
    ordered: list[str] = []
    for item in results:
        target = str(item.get("target", "")).strip()
        if target and target not in ordered:
            ordered.append(target)
    if "orchid" in ordered and "vllm" in ordered:
        return ["orchid", "vllm"]
    return ordered[:2]


def _write_compare_report(results: list[dict[str, Any]], out_dir: Path) -> Path | None:
    compare_targets = _pick_compare_targets(results)
    if len(compare_targets) < 2:
        return None
    lhs, rhs = compare_targets[0], compare_targets[1]
    by_scenario: dict[str, dict[str, dict[str, Any]]] = {}
    for item in results:
        target = str(item.get("target", ""))
        scenario = str(item.get("scenario", {}).get("name", ""))
        if not target or not scenario:
            continue
        by_scenario.setdefault(scenario, {})[target] = item

    lines = [
        "# EvalScope Compare Summary",
        "",
        f"- left: `{lhs}`",
        f"- right: `{rhs}`",
        "",
        f"| scenario | {lhs} req/s | {rhs} req/s | {lhs}/{rhs} req ratio | {lhs} tok/s | {rhs} tok/s | {lhs}/{rhs} tok ratio | {lhs} TTFT(s) | {rhs} TTFT(s) | {lhs} TPOT(s) | {rhs} TPOT(s) |",
        f"| - | -: | -: | -: | -: | -: | -: | -: | -: | -: | -: |",
    ]
    shared_scenarios = [name for name, targets in by_scenario.items() if lhs in targets and rhs in targets]
    for scenario in sorted(shared_scenarios):
        left_metrics = by_scenario[scenario][lhs].get("summary_metrics") or {}
        right_metrics = by_scenario[scenario][rhs].get("summary_metrics") or {}
        left_req = _to_float(left_metrics.get("Request throughput (req/s)"))
        right_req = _to_float(right_metrics.get("Request throughput (req/s)"))
        left_tok = _to_float(left_metrics.get("Output token throughput (tok/s)"))
        right_tok = _to_float(right_metrics.get("Output token throughput (tok/s)"))
        left_ttft = _to_float(left_metrics.get("Average time to first token (s)"))
        right_ttft = _to_float(right_metrics.get("Average time to first token (s)"))
        left_tpot = _to_float(left_metrics.get("Average time per output token (s)"))
        right_tpot = _to_float(right_metrics.get("Average time per output token (s)"))
        req_ratio = (left_req / right_req) if left_req is not None and right_req not in (None, 0.0) else None
        tok_ratio = (left_tok / right_tok) if left_tok is not None and right_tok not in (None, 0.0) else None
        lines.append(
            "| "
            + " | ".join(
                [
                    scenario,
                    "" if left_req is None else f"{left_req:.2f}",
                    "" if right_req is None else f"{right_req:.2f}",
                    "" if req_ratio is None else f"{req_ratio:.4f}",
                    "" if left_tok is None else f"{left_tok:.2f}",
                    "" if right_tok is None else f"{right_tok:.2f}",
                    "" if tok_ratio is None else f"{tok_ratio:.4f}",
                    "" if left_ttft is None else f"{left_ttft:.3f}",
                    "" if right_ttft is None else f"{right_ttft:.3f}",
                    "" if left_tpot is None else f"{left_tpot:.3f}",
                    "" if right_tpot is None else f"{right_tpot:.3f}",
                ]
            )
            + " |"
        )
    if not shared_scenarios:
        lines.append("- no shared scenarios found across selected targets")

    report_path = out_dir / "compare_summary.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def _load_results_from_meta(paths: list[str]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    for raw in paths:
        meta_path = Path(raw)
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
        for item in payload.get("results", []):
            merged.append(item)
    return merged


def _parse_target(raw: str) -> tuple[str, str]:
    if "=" not in raw:
        raise argparse.ArgumentTypeError("target must be in NAME=URL format")
    name, url = raw.split("=", 1)
    name = name.strip()
    url = url.strip()
    if not name or not url:
        raise argparse.ArgumentTypeError("target must be in NAME=URL format")
    return name, url


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", choices=["smoke", "standard", "full"], default="smoke")
    ap.add_argument("--target", action="append", default=[], help="NAME=URL")
    ap.add_argument("--spawn-vllm-target", default=None)
    ap.add_argument("--spawn-orchid-target", default=None)
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--vllm-model", default=None)
    ap.add_argument("--tokenizer-path", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--vllm-port", type=int, default=8801)
    ap.add_argument("--orchid-port", type=int, default=8001)
    ap.add_argument("--orchid-src", default="/workspace/torchpipe/plugins/orchid/src")
    ap.add_argument("--orchid-model-path", default="/root/.cache/orchid/models/Qwen_Qwen3-0.6B/fp16/model.composite.onnx")
    ap.add_argument("--orchid-engine-path", default="/root/.cache/orchid/models/Qwen_Qwen3-0.6B/fp16/model.rtx5070ti.trt1016.plan")
    ap.add_argument("--orchid-page-size", type=int, default=16)
    ap.add_argument("--orchid-max-pages", type=int, default=32768)
    ap.add_argument("--orchid-kv-cache-reserved-mb", type=int, default=4096)
    ap.add_argument("--orchid-gpu-mem", type=float, default=0.85)
    ap.add_argument("--server-timeout-s", type=float, default=180.0)
    ap.add_argument("--vllm-max-model-len", type=int, default=4096)
    ap.add_argument("--vllm-gpu-mem", type=float, default=0.35)
    ap.add_argument("--vllm-enforce-eager", action="store_true")
    ap.add_argument("--out-dir", default=str(_default_out_dir() / time.strftime("%Y%m%d_%H%M%S")))
    ap.add_argument("--print-commands", action="store_true")
    ap.add_argument("--merge-meta", action="append", default=[])
    ns = ap.parse_args()
    ns.extra_args = {"ignore_eos": True}

    if ns.merge_meta:
        results = _load_results_from_meta(ns.merge_meta)
        report_path = _write_report(results, ns)
        compare_path = _write_compare_report(results, Path(ns.out_dir))
        print(json.dumps({"out_dir": ns.out_dir, "summary": str(report_path), "compare_summary": str(compare_path) if compare_path else None}, ensure_ascii=False))
        return

    targets: list[tuple[str, str]] = [_parse_target(raw) for raw in ns.target]
    spawned: list[subprocess.Popen[str]] = []
    if ns.spawn_vllm_target:
        proc, url = _spawn_vllm(ns, ns.spawn_vllm_target)
        spawned.append(proc)
        targets.append((ns.spawn_vllm_target, url))
    if ns.spawn_orchid_target:
        proc, url = _spawn_orchid(ns, ns.spawn_orchid_target)
        spawned.append(proc)
        targets.append((ns.spawn_orchid_target, url))
    if not targets:
        raise SystemExit("at least one --target NAME=URL, --spawn-vllm-target NAME, or --spawn-orchid-target NAME is required")

    _ensure_binary("evalscope")
    scenarios = _scenarios_for_suite(ns.suite)
    results: list[dict[str, Any]] = []
    try:
        for target_name, url in targets:
            for scenario in scenarios:
                result = _run_target_scenario(target_name, url, scenario, ns)
                results.append(result)
                if ns.print_commands:
                    print(shlex.join(result["command"]))
                if int(result["returncode"]) != 0:
                    raise SystemExit(f"evalscope perf failed: target={target_name} scenario={scenario.name}")
    finally:
        for proc in spawned:
            try:
                proc.terminate()
                proc.wait(timeout=10)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass

    report_path = _write_report(results, ns)
    compare_path = _write_compare_report(results, Path(ns.out_dir))
    print(json.dumps({"out_dir": ns.out_dir, "summary": str(report_path), "compare_summary": str(compare_path) if compare_path else None, "targets": [name for name, _ in targets]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
