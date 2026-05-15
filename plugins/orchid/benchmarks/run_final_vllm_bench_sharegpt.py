import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen

from fetch_sharegpt_vllm import DEFAULT_OUT as DEFAULT_SHAREGPT_DATASET_PATH
from fetch_sharegpt_vllm import DEFAULT_SHAREGPT_URL
from fetch_sharegpt_vllm import ensure_sharegpt_dataset
from orchid.paths import benchmark_artifact


def _ensure_dir(path: str) -> None:
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)


def _now_tag() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())


def _http_get_json(url: str, timeout_s: float) -> dict[str, Any]:
    req = Request(url, method="GET")
    with urlopen(req, timeout=float(timeout_s)) as r:
        data = r.read().decode("utf-8")
        return json.loads(data)

def _http_post_json(url: str, payload: dict[str, Any], timeout_s: float) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    req = Request(url, method="POST", data=body, headers={"Content-Type": "application/json"})
    with urlopen(req, timeout=float(timeout_s)) as r:
        data = r.read().decode("utf-8")
        return json.loads(data)


def _wait_ready(host: str, port: int, *, timeout_s: float, model: str) -> None:
    t0 = time.time()
    last_err = None
    url_models = f"http://{host}:{int(port)}/v1/models"
    url_chat = f"http://{host}:{int(port)}/v1/chat/completions"
    payload = {"model": str(model), "messages": [{"role": "user", "content": "ping"}], "max_tokens": 1, "temperature": 0}
    while True:
        if time.time() - t0 > float(timeout_s):
            raise TimeoutError(f"Server not ready: {url_chat} last_err={last_err}")
        try:
            try:
                _ = _http_get_json(url_models, timeout_s=2.0)
                return
            except Exception:
                _ = _http_post_json(url_chat, payload, timeout_s=5.0)
            return
        except (URLError, TimeoutError, json.JSONDecodeError) as e:
            last_err = repr(e)
            time.sleep(0.2)


def _terminate_proc(proc: subprocess.Popen, *, timeout_s: float, allow_kill: bool) -> None:
    if proc.poll() is not None:
        return
    try:
        proc.send_signal(signal.SIGTERM)
    except Exception:
        try:
            proc.terminate()
        except Exception:
            return
    t0 = time.time()
    while proc.poll() is None and (time.time() - t0) < float(timeout_s):
        time.sleep(0.2)
    if proc.poll() is None and bool(allow_kill):
        try:
            proc.kill()
        except Exception:
            pass


def _parse_vllm_bench_text(text: str) -> dict[str, Any]:
    out: dict[str, Any] = {}

    def find_float(pattern: str, key: str) -> None:
        m = re.search(pattern, text)
        if m:
            try:
                out[key] = float(m.group(1))
            except Exception:
                pass

    find_float(r"Output tok/s\s*\|\s*([0-9]*\.?[0-9]+)", "output_tok_s")
    find_float(r"Mean TTFT \(ms\)\s*\|\s*([0-9]*\.?[0-9]+)", "mean_ttft_ms")
    find_float(r"Mean TPOT \(ms\)\s*\|\s*([0-9]*\.?[0-9]+)", "mean_tpot_ms")
    find_float(r"Output token throughput \(tok/s\):\s*([0-9]*\.?[0-9]+)", "output_tok_s")
    find_float(r"Request throughput \(req/s\):\s*([0-9]*\.?[0-9]+)", "request_rps")
    find_float(r"Mean TTFT \(ms\):\s*([0-9]*\.?[0-9]+)", "mean_ttft_ms")
    find_float(r"Mean TPOT \(ms\):\s*([0-9]*\.?[0-9]+)", "mean_tpot_ms")
    find_float(r"P99 TTFT \(ms\):\s*([0-9]*\.?[0-9]+)", "p99_ttft_ms")
    find_float(r"P99 TPOT \(ms\):\s*([0-9]*\.?[0-9]+)", "p99_tpot_ms")

    m = re.search(r"Output tok/s:\s*([0-9]*\.?[0-9]+)", text)
    if m:
        try:
            out["output_tok_s"] = float(m.group(1))
        except Exception:
            pass
    m = re.search(r"Mean TTFT:\s*([0-9]*\.?[0-9]+)", text)
    if m:
        try:
            out["mean_ttft_ms"] = float(m.group(1))
        except Exception:
            pass
    m = re.search(r"Mean TPOT:\s*([0-9]*\.?[0-9]+)", text)
    if m:
        try:
            out["mean_tpot_ms"] = float(m.group(1))
        except Exception:
            pass
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--tokenizer-path", required=True)
    ap.add_argument("--engine-path", required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8002)
    ap.add_argument("--server-app", default="orchid.llmscheduler.server.api_server:app")
    ap.add_argument("--server-timeout-s", type=float, default=120.0)
    ap.add_argument("--shutdown-timeout-s", type=float, default=20.0)
    ap.add_argument("--allow-kill", action="store_true")

    ap.add_argument("--dataset-name", default="sharegpt")
    ap.add_argument("--dataset-path", default=str(DEFAULT_SHAREGPT_DATASET_PATH))
    ap.add_argument("--dataset-url", default=str(DEFAULT_SHAREGPT_URL))
    ap.add_argument("--no-auto-download-dataset", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--sharegpt-output-len", type=int, default=128)
    ap.add_argument("--random-input-len", type=int, default=512)
    ap.add_argument("--random-output-len", type=int, default=128)
    ap.add_argument("--num-prompts", type=int, default=200)
    ap.add_argument("--num-warmups", type=int, default=10)
    ap.add_argument("--request-rate", default="inf")
    ap.add_argument("--max-concurrency", type=int, default=32)

    ap.add_argument("--out-dir", default=benchmark_artifact("final_vllm_bench", "sharegpt"))
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    dataset_path = str(args.dataset_path).strip()
    if str(args.dataset_name) == "sharegpt":
        dataset_candidates = [
            dataset_path,
            str(DEFAULT_SHAREGPT_DATASET_PATH),
            "/root/.cache/orchid/datasets/sharegpt/ShareGPT_V3_unfiltered_cleaned_split.json",
        ]
        dataset_candidates = [x for x in dataset_candidates if x]
        dataset_path = ""
        for p in dataset_candidates:
            if os.path.exists(p) and os.path.getsize(p) > 0:
                dataset_path = str(p)
                break
        if not dataset_path:
            if bool(args.no_auto_download_dataset):
                raise FileNotFoundError(f"ShareGPT dataset not found in candidates: {dataset_candidates}")
            dataset_path = ensure_sharegpt_dataset(dataset_candidates[0], url=str(args.dataset_url))

    tag = str(args.tag).strip() or _now_tag()
    out_dir = os.path.abspath(os.path.join(str(args.out_dir), str(tag)))
    os.makedirs(out_dir, exist_ok=True)

    server_log = os.path.join(out_dir, "server.log")
    bench_log = os.path.join(out_dir, "vllm_bench.log")
    meta_path = os.path.join(out_dir, "meta.json")
    summary_path = os.path.join(out_dir, "summary.md")

    env = dict(os.environ)
    env["MODEL_PATH"] = str(args.model_path)
    env["TOKENIZER_PATH"] = str(args.tokenizer_path)
    env["ENGINE_PATH"] = str(args.engine_path)
    env["LLMSCHEDULER_TEST_MODE"] = "0"
    env.setdefault("LLMSCHEDULER_APPLY_CHAT_TEMPLATE", "1")
    env.setdefault("LLMSCHEDULER_TRT_USE_TORCH_STREAM", "0")
    env.setdefault("LLMSCHEDULER_MAX_BATCH_TOKENS", "4096")

    env.setdefault("FLASHINFER_DISABLE_VERSION_CHECK", "1")
    env.setdefault("LLMSCHEDULER_FLASHINFER_WORKSPACE_MB", "512")
    env.setdefault("LLMSCHEDULER_FLASHINFER_DECODE_CUDAGRAPH", "0")
    env.setdefault("LLMSCHEDULER_FLASHINFER_MAX_BATCH_SIZE", str(int(args.max_concurrency)))
    env.setdefault("LLMSCHEDULER_FLASHINFER_DISABLE_SPLIT_KV", "1")

    env.setdefault("LLMSCHEDULER_TRT_CUDAGRAPH", "1")
    env.setdefault("LLMSCHEDULER_TRT_CUDAGRAPH_CTX_KEY", "page_and_last")
    env.setdefault("LLMSCHEDULER_TRT_CUDAGRAPH_LAST_BUCKET", "8")
    env.setdefault("LLMSCHEDULER_TRT_CUDAGRAPH_MAX_GRAPHS", "8")
    env.setdefault("LLMSCHEDULER_TRT_CUDAGRAPH_FALLBACK", "1")

    server_cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        str(args.server_app),
        "--host",
        str(args.host),
        "--port",
        str(int(args.port)),
        "--no-access-log",
        "--log-level",
        "warning",
    ]

    bench_cmd = [
        "vllm",
        "bench",
        "serve",
        "--backend",
        "openai-chat",
        "--host",
        str(args.host),
        "--port",
        str(int(args.port)),
        "--endpoint",
        "/v1/chat/completions",
        "--model",
        str(args.tokenizer_path),
        "--tokenizer",
        str(args.tokenizer_path),
        "--dataset-name",
        str(args.dataset_name),
        "--num-prompts",
        str(int(args.num_prompts)),
        "--num-warmups",
        str(int(args.num_warmups)),
        "--request-rate",
        str(args.request_rate),
        "--max-concurrency",
        str(int(args.max_concurrency)),
        "--disable-tqdm",
        "--skip-chat-template",
        "--temperature",
        "0",
    ]
    if str(args.dataset_name) == "sharegpt":
        bench_cmd.extend(["--dataset-path", str(dataset_path)])
        bench_cmd.extend(["--sharegpt-output-len", str(int(args.sharegpt_output_len))])
    if str(args.dataset_name) == "random":
        bench_cmd.extend(["--random-input-len", str(int(args.random_input_len)), "--random-output-len", str(int(args.random_output_len))])

    if bool(args.dry_run):
        sys.stdout.write(json.dumps({"dataset_path": dataset_path, "server_cmd": server_cmd, "bench_cmd": bench_cmd, "out_dir": out_dir}, ensure_ascii=False))
        return

    meta = {
        "ts": int(time.time()),
        "out_dir": out_dir,
        "server_cmd": server_cmd,
        "bench_cmd": bench_cmd,
        "env_subset": {k: env.get(k) for k in sorted(env.keys()) if k.startswith("LLMSCHEDULER_") or k.startswith("FLASHINFER_") or k in ("MODEL_PATH", "TOKENIZER_PATH", "ENGINE_PATH")},
    }
    _ensure_dir(meta_path)
    with open(meta_path, "w") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    with open(server_log, "w") as sf:
        proc = subprocess.Popen(server_cmd, stdout=sf, stderr=subprocess.STDOUT, env=env)
        try:
            _wait_ready(str(args.host), int(args.port), timeout_s=float(args.server_timeout_s), model=str(args.tokenizer_path))
            t0 = time.perf_counter()
            with open(bench_log, "w") as bf:
                r = subprocess.run(bench_cmd, stdout=bf, stderr=subprocess.STDOUT, text=True)
            dt = float(time.perf_counter() - t0)
            with open(bench_log, "r") as bf:
                bench_text = bf.read()
            metrics = _parse_vllm_bench_text(bench_text)
            _terminate_proc(proc, timeout_s=float(args.shutdown_timeout_s), allow_kill=bool(args.allow_kill))

            with open(summary_path, "w") as f:
                f.write("# vLLM bench serve (ShareGPT) summary\n\n")
                f.write(f"- out_dir: {os.path.relpath(out_dir, os.getcwd())}\n")
                f.write(f"- wall_s: {dt:.3f}\n")
                f.write(f"- server_log: {os.path.basename(server_log)}\n")
                f.write(f"- bench_log: {os.path.basename(bench_log)}\n")
                f.write(f"- meta: {os.path.basename(meta_path)}\n\n")
                f.write("|metric|value|\n")
                f.write("|-:|-:|\n")
                for k in ("request_rps", "output_tok_s", "mean_ttft_ms", "p99_ttft_ms", "mean_tpot_ms", "p99_tpot_ms"):
                    if k in metrics:
                        f.write(f"|{k}|{metrics[k]}|\n")
                f.write("\n")
                f.write("## Env subset\n\n")
                for k, v in meta["env_subset"].items():
                    f.write(f"- {k}={v}\n")

            sys.stdout.write(json.dumps({"out_dir": out_dir, "meta": meta_path, "summary": summary_path}, ensure_ascii=False))
            sys.exit(int(r.returncode))
        finally:
            _terminate_proc(proc, timeout_s=float(args.shutdown_timeout_s), allow_kill=bool(args.allow_kill))


if __name__ == "__main__":
    main()
