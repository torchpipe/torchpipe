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


def _http_post_json(url: str, payload: dict[str, Any], timeout_s: float) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    req = Request(url, method="POST", data=body, headers={"Content-Type": "application/json"})
    with urlopen(req, timeout=float(timeout_s)) as r:
        data = r.read().decode("utf-8")
        return json.loads(data)


def _wait_ready(host: str, port: int, *, timeout_s: float, model: str) -> None:
    t0 = time.time()
    last_err = None
    url_chat = f"http://{host}:{int(port)}/v1/chat/completions"
    payload = {"model": str(model), "messages": [{"role": "user", "content": "ping"}], "max_tokens": 1, "temperature": 0}
    while True:
        if time.time() - t0 > float(timeout_s):
            raise TimeoutError(f"Server not ready: {url_chat} last_err={last_err}")
        try:
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

    find_float(r"Request throughput \(req/s\):\s*([0-9]*\.?[0-9]+)", "request_rps")
    find_float(r"Output token throughput \(tok/s\):\s*([0-9]*\.?[0-9]+)", "output_tok_s")
    find_float(r"Mean TTFT \(ms\):\s*([0-9]*\.?[0-9]+)", "mean_ttft_ms")
    find_float(r"P99 TTFT \(ms\):\s*([0-9]*\.?[0-9]+)", "p99_ttft_ms")
    find_float(r"Mean TPOT \(ms\):\s*([0-9]*\.?[0-9]+)", "mean_tpot_ms")
    find_float(r"P99 TPOT \(ms\):\s*([0-9]*\.?[0-9]+)", "p99_tpot_ms")
    find_float(r"Failed requests:\s*([0-9]*\.?[0-9]+)", "failed_reqs")
    return out


def _run_one(
    *,
    out_dir: str,
    server_kind: str,
    server_app: str,
    host: str,
    port: int,
    allow_kill: bool,
    server_timeout_s: float,
    shutdown_timeout_s: float,
    bench_cmd: list[str],
    env: dict[str, str],
    ready_model: str,
    server_extra_args: list[str] | None = None,
) -> dict[str, Any]:
    os.makedirs(out_dir, exist_ok=True)
    server_log = os.path.join(out_dir, f"{server_kind}_server.log")
    bench_log = os.path.join(out_dir, f"{server_kind}_bench.log")

    if server_kind == "llmscheduler":
        server_cmd = [
            sys.executable,
            "-m",
            "uvicorn",
            str(server_app),
            "--host",
            str(host),
            "--port",
            str(int(port)),
            "--no-access-log",
            "--log-level",
            "warning",
        ]
    elif server_kind == "vllm":
        server_cmd = [
            sys.executable,
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            str(ready_model),
            "--tokenizer",
            str(ready_model),
            "--host",
            str(host),
            "--port",
            str(int(port)),
            "--dtype",
            "float16",
            "--tensor-parallel-size",
            "1",
            "--no-enable-log-requests",
        ]
    elif server_kind == "sglang":
        server_cmd = [
            sys.executable,
            "-m",
            "sglang.launch_server",
            "--model-path",
            str(ready_model),
            "--tokenizer-path",
            str(ready_model),
            "--trust-remote-code",
            "--host",
            str(host),
            "--port",
            str(int(port)),
            "--dtype",
            "float16",
        ]
    else:
        raise ValueError(f"Unknown server_kind={server_kind}")
    if server_extra_args:
        server_cmd.extend([str(x) for x in server_extra_args])

    t0 = time.perf_counter()
    with open(server_log, "w") as sf:
        proc = subprocess.Popen(server_cmd, stdout=sf, stderr=subprocess.STDOUT, env=env)
        try:
            _wait_ready(str(host), int(port), timeout_s=float(server_timeout_s), model=str(ready_model))
            with open(bench_log, "w") as bf:
                r = subprocess.run(bench_cmd, stdout=bf, stderr=subprocess.STDOUT, text=True, env=env)
            wall_s = float(time.perf_counter() - t0)
            with open(bench_log, "r") as bf:
                bench_text = bf.read()
            metrics = _parse_vllm_bench_text(bench_text)
            return {"returncode": int(r.returncode), "wall_s": wall_s, "server_cmd": server_cmd, "bench_cmd": bench_cmd, "metrics": metrics}
        finally:
            _terminate_proc(proc, timeout_s=float(shutdown_timeout_s), allow_kill=bool(allow_kill))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--tokenizer-path", required=True)
    ap.add_argument("--engine-path", required=True)
    ap.add_argument("--vllm-model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--server-app", default="orchid.llmscheduler.server.api_server:app")

    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port-base", type=int, default=8100)
    ap.add_argument("--server-timeout-s", type=float, default=180.0)
    ap.add_argument("--shutdown-timeout-s", type=float, default=30.0)
    ap.add_argument("--allow-kill", action="store_true")

    ap.add_argument("--concurrency", default="1,4,8,10")
    ap.add_argument("--num-prompts", type=int, default=200)
    ap.add_argument("--num-warmups", type=int, default=10)
    ap.add_argument("--request-rate", default="inf")
    ap.add_argument("--sharegpt-output-len", type=int, default=128)
    ap.add_argument("--dataset-path", default=str(DEFAULT_SHAREGPT_DATASET_PATH))
    ap.add_argument("--dataset-url", default=str(DEFAULT_SHAREGPT_URL))
    ap.add_argument("--no-auto-download-dataset", action="store_true")
    ap.add_argument("--compare-vllm-eager", action="store_true")
    ap.add_argument("--vllm-enforce-eager", action="store_true")
    ap.add_argument("--vllm-max-model-len", type=int, default=4096)
    ap.add_argument("--vllm-gpu-mem", type=float, default=0.25)
    ap.add_argument("--dry-run", action="store_true")

    ap.add_argument("--tag", default="")
    ap.add_argument("--out-dir", default=benchmark_artifact("gap_sharegpt"))
    ap.add_argument("--include-sglang", action="store_true")
    args = ap.parse_args()
    dataset_candidates = [
        str(args.dataset_path).strip(),
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

    conc = [int(x.strip()) for x in str(args.concurrency).split(",") if x.strip()]
    conc = [int(x) for x in conc if int(x) > 0]
    conc.sort()
    if not conc:
        raise ValueError("--concurrency must contain at least one positive integer")

    env_base = dict(os.environ)
    env_base["MODEL_PATH"] = str(args.model_path)
    env_base["TOKENIZER_PATH"] = str(args.tokenizer_path)
    env_base["ENGINE_PATH"] = str(args.engine_path)
    env_base["LLMSCHEDULER_TEST_MODE"] = "0"
    env_base.setdefault("LLMSCHEDULER_APPLY_CHAT_TEMPLATE", "1")
    env_base.setdefault("LLMSCHEDULER_TRT_USE_TORCH_STREAM", "0")
    env_base.setdefault("LLMSCHEDULER_MAX_BATCH_TOKENS", "4096")

    env_base.setdefault("FLASHINFER_DISABLE_VERSION_CHECK", "1")
    env_base.setdefault("LLMSCHEDULER_FLASHINFER_WORKSPACE_MB", "512")
    env_base.setdefault("LLMSCHEDULER_FLASHINFER_DECODE_CUDAGRAPH", "0")
    env_base.setdefault("LLMSCHEDULER_FLASHINFER_DISABLE_SPLIT_KV", "1")

    env_base.setdefault("LLMSCHEDULER_TRT_CUDAGRAPH", "1")
    env_base.setdefault("LLMSCHEDULER_TRT_CUDAGRAPH_CTX_KEY", "page_and_last")
    env_base.setdefault("LLMSCHEDULER_TRT_CUDAGRAPH_LAST_BUCKET", "8")
    env_base.setdefault("LLMSCHEDULER_TRT_CUDAGRAPH_MAX_GRAPHS", "8")
    env_base.setdefault("LLMSCHEDULER_TRT_CUDAGRAPH_FALLBACK", "1")

    results: list[dict[str, Any]] = []

    for i, c in enumerate(conc):
        vllm_server_common_args = [
            "--max-model-len",
            str(int(args.vllm_max_model_len)),
            "--gpu-memory-utilization",
            str(float(args.vllm_gpu_mem)),
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
            str(int(args.port_base) + int(i)),
            "--endpoint",
            "/v1/chat/completions",
            "--model",
            str(args.vllm_model),
            "--tokenizer",
            str(args.vllm_model),
            "--dataset-name",
            "sharegpt",
            "--dataset-path",
            str(dataset_path),
            "--sharegpt-output-len",
            str(int(args.sharegpt_output_len)),
            "--num-prompts",
            str(int(args.num_prompts)),
            "--num-warmups",
            str(int(args.num_warmups)),
            "--request-rate",
            str(args.request_rate),
            "--max-concurrency",
            str(int(c)),
            "--disable-tqdm",
            "--skip-chat-template",
            "--temperature",
            "0",
        ]

        env = dict(env_base)
        env["LLMSCHEDULER_FLASHINFER_MAX_BATCH_SIZE"] = str(int(c))
        if bool(args.dry_run):
            ours_server_cmd = [sys.executable, "-m", "uvicorn", str(args.server_app), "--host", str(args.host), "--port", str(int(args.port_base) + int(i))]
            vllm_server_cmd = [
                sys.executable,
                "-m",
                "vllm.entrypoints.openai.api_server",
                "--model",
                str(args.vllm_model),
                "--tokenizer",
                str(args.vllm_model),
                "--host",
                str(args.host),
                "--port",
                str(int(args.port_base) + int(i) + 100),
                "--dtype",
                "float16",
                "--tensor-parallel-size",
                "1",
                "--no-enable-log-requests",
            ]
            vllm_server_cmd.extend(vllm_server_common_args)
            if bool(args.vllm_enforce_eager) and not bool(args.compare_vllm_eager):
                vllm_server_cmd.append("--enforce-eager")
            out = {
                "dataset_path": dataset_path,
                "concurrency": conc,
                "first_concurrency": int(c),
                "llmscheduler_server_cmd": ours_server_cmd,
                "vllm_server_cmd": vllm_server_cmd,
                "bench_cmd_llmscheduler": bench_cmd,
                "bench_cmd_vllm": [*bench_cmd[:], "--port", str(int(args.port_base) + int(i) + 100)],
                "compare_vllm_eager": bool(args.compare_vllm_eager),
            }
            sys.stdout.write(json.dumps(out, ensure_ascii=False))
            return

        ours = _run_one(
            out_dir=os.path.join(out_dir, f"conc_{c}", "llmscheduler"),
            server_kind="llmscheduler",
            server_app=str(args.server_app),
            host=str(args.host),
            port=int(args.port_base) + int(i),
            allow_kill=bool(args.allow_kill),
            server_timeout_s=float(args.server_timeout_s),
            shutdown_timeout_s=float(args.shutdown_timeout_s),
            bench_cmd=bench_cmd,
            env=env,
            ready_model=str(args.vllm_model),
        )
        vllm_default = _run_one(
            out_dir=os.path.join(out_dir, f"conc_{c}", "vllm"),
            server_kind="vllm",
            server_app=str(args.server_app),
            host=str(args.host),
            port=int(args.port_base) + int(i) + 100,
            allow_kill=bool(args.allow_kill),
            server_timeout_s=float(args.server_timeout_s),
            shutdown_timeout_s=float(args.shutdown_timeout_s),
            bench_cmd=[*bench_cmd[:], "--port", str(int(args.port_base) + int(i) + 100)],
            env=env,
            ready_model=str(args.vllm_model),
            server_extra_args=[
                *vllm_server_common_args,
                *(["--enforce-eager"] if bool(args.vllm_enforce_eager) and not bool(args.compare_vllm_eager) else []),
            ],
        )
        rrow: dict[str, Any] = {"concurrency": int(c), "llmscheduler": ours, "vllm": vllm_default}
        if bool(args.compare_vllm_eager):
            vllm_eager = _run_one(
                out_dir=os.path.join(out_dir, f"conc_{c}", "vllm_eager"),
                server_kind="vllm",
                server_app=str(args.server_app),
                host=str(args.host),
                port=int(args.port_base) + int(i) + 101,
                allow_kill=bool(args.allow_kill),
                server_timeout_s=float(args.server_timeout_s),
                shutdown_timeout_s=float(args.shutdown_timeout_s),
                bench_cmd=[*bench_cmd[:], "--port", str(int(args.port_base) + int(i) + 101)],
                env=env,
                ready_model=str(args.vllm_model),
                server_extra_args=[*vllm_server_common_args, "--enforce-eager"],
            )
            rrow["vllm_eager"] = vllm_eager
        results.append(rrow)
        if bool(args.include_sglang):
            sg = _run_one(
                out_dir=os.path.join(out_dir, f"conc_{c}", "sglang"),
                server_kind="sglang",
                server_app=str(args.server_app),
                host=str(args.host),
                port=int(args.port_base) + int(i) + 200,
                allow_kill=bool(args.allow_kill),
                server_timeout_s=float(args.server_timeout_s),
                shutdown_timeout_s=float(args.shutdown_timeout_s),
                bench_cmd=[*bench_cmd[:], "--port", str(int(args.port_base) + int(i) + 200)],
                env=env,
                ready_model=str(args.vllm_model),
            )
            results[-1]["sglang"] = sg

    meta_path = os.path.join(out_dir, "meta.json")
    with open(meta_path, "w") as f:
        json.dump(
            {
                "ts": int(time.time()),
                "tag": tag,
                "concurrency": conc,
                "model_path": str(args.model_path),
                "tokenizer_path": str(args.tokenizer_path),
                "engine_path": str(args.engine_path),
                "vllm_model": str(args.vllm_model),
                "env_subset": {k: env_base.get(k) for k in sorted(env_base.keys()) if k.startswith("LLMSCHEDULER_") or k.startswith("FLASHINFER_")},
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    gap_md = os.path.join(out_dir, "gap_summary.md")
    with open(gap_md, "w") as f:
        f.write("# ShareGPT gap summary (llmscheduler vs vLLM)\n\n")
        f.write(f"- out_dir: {os.path.relpath(out_dir, os.getcwd())}\n")
        f.write(f"- meta: {os.path.basename(meta_path)}\n\n")
        f.write("|conc|ours tok/s|vllm tok/s|ratio ours/vllm|ours TTFT|vllm TTFT|ours TPOT|vllm TPOT|ours failed|vllm failed|\n")
        f.write("|-:|-:|-:|-:|-:|-:|-:|-:|-:|-:|\n")
        for r in results:
            c = int(r["concurrency"])
            om = dict(r["llmscheduler"]["metrics"] or {})
            vm = dict(r["vllm"]["metrics"] or {})
            ours_tok = float(om.get("output_tok_s") or 0.0)
            vllm_tok = float(vm.get("output_tok_s") or 0.0)
            ratio = (ours_tok / vllm_tok) if vllm_tok > 0 else 0.0
            f.write(
                f"|{c}|{ours_tok:.2f}|{vllm_tok:.2f}|{ratio:.4f}|{float(om.get('mean_ttft_ms') or 0.0):.2f}|{float(vm.get('mean_ttft_ms') or 0.0):.2f}|{float(om.get('mean_tpot_ms') or 0.0):.2f}|{float(vm.get('mean_tpot_ms') or 0.0):.2f}|{int(om.get('failed_reqs') or 0)}|{int(vm.get('failed_reqs') or 0)}|\n"
            )
        if bool(args.compare_vllm_eager):
            f.write("\n## vLLM enforce-eager (CUDA Graph off)\n\n")
            f.write("|conc|vllm eager tok/s|vllm tok/s|ratio eager/default|eager TTFT|default TTFT|eager TPOT|default TPOT|\n")
            f.write("|-:|-:|-:|-:|-:|-:|-:|-:|\n")
            for r in results:
                if "vllm_eager" not in r:
                    continue
                c = int(r["concurrency"])
                em = dict((r["vllm_eager"] or {}).get("metrics") or {})
                dm = dict((r["vllm"] or {}).get("metrics") or {})
                e_tok = float(em.get("output_tok_s") or 0.0)
                d_tok = float(dm.get("output_tok_s") or 0.0)
                ratio_ed = (e_tok / d_tok) if d_tok > 0 else 0.0
                f.write(
                    f"|{c}|{e_tok:.2f}|{d_tok:.2f}|{ratio_ed:.4f}|{float(em.get('mean_ttft_ms') or 0.0):.2f}|{float(dm.get('mean_ttft_ms') or 0.0):.2f}|{float(em.get('mean_tpot_ms') or 0.0):.2f}|{float(dm.get('mean_tpot_ms') or 0.0):.2f}|\n"
                )
        if bool(args.include_sglang):
            f.write("\n## sglang\n\n")
            f.write("|conc|sglang tok/s|ours tok/s|ratio ours/sglang|sglang TTFT|ours TTFT|\n")
            f.write("|-:|-:|-:|-:|-:|-:|\n")
            for r in results:
                c = int(r["concurrency"])
                if "sglang" not in r:
                    continue
                sm = dict((r["sglang"] or {}).get("metrics") or {})
                om = dict((r["llmscheduler"] or {}).get("metrics") or {})
                s_tok = float(sm.get("output_tok_s") or 0.0)
                o_tok = float(om.get("output_tok_s") or 0.0)
                ratio2 = (o_tok / s_tok) if s_tok > 0 else 0.0
                f.write(f"|{c}|{s_tok:.2f}|{o_tok:.2f}|{ratio2:.4f}|{float(sm.get('mean_ttft_ms') or 0.0):.2f}|{float(om.get('mean_ttft_ms') or 0.0):.2f}|\n")

    sys.stdout.write(json.dumps({"out_dir": out_dir, "meta": meta_path, "gap_summary": gap_md}, ensure_ascii=False))


if __name__ == "__main__":
    main()
