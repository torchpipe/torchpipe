import argparse
import json
import os
import sys
import time
from urllib.error import URLError
from urllib.request import Request, urlopen

from orchid.paths import benchmark_dataset


DEFAULT_SHAREGPT_URL = "https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json"
DEFAULT_OUT = benchmark_dataset("ShareGPT_V3_unfiltered_cleaned_split.json")


def _ensure_dir(path: str) -> None:
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)


def _validate_sharegpt_json(path: str) -> None:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list) or not data:
        raise ValueError("ShareGPT json must be a non-empty list")
    x = data[0]
    if not isinstance(x, dict):
        raise ValueError("ShareGPT json list elements must be dicts")
    conv = x.get("conversations")
    if not isinstance(conv, list) or not conv:
        raise ValueError("ShareGPT entry must have non-empty conversations list")


def ensure_sharegpt_dataset(
    out_path: str,
    *,
    url: str = DEFAULT_SHAREGPT_URL,
    timeout_s: float = 600.0,
    retries: int = 5,
    validate: bool = True,
) -> str:
    out_path = os.path.abspath(str(out_path))
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        if validate:
            _validate_sharegpt_json(out_path)
        return out_path

    _ensure_dir(out_path)
    tmp_path = out_path + ".tmp"

    last_err: Exception | None = None
    for attempt in range(int(retries) + 1):
        if attempt:
            time.sleep(min(10.0, 1.0 * (2**(attempt - 1))))
        try:
            existing = 0
            if os.path.exists(tmp_path):
                try:
                    existing = int(os.path.getsize(tmp_path))
                except Exception:
                    existing = 0

            headers = {"User-Agent": "onnxscheduler-bench/1.0"}
            if existing > 0:
                headers["Range"] = f"bytes={existing}-"
            req = Request(str(url), method="GET", headers=headers)

            with urlopen(req, timeout=float(timeout_s)) as r:
                code = getattr(r, "status", None) or r.getcode()
                mode = "ab" if (existing > 0 and int(code) == 206) else "wb"
                total = existing if mode == "ab" else 0
                if mode == "wb" and existing > 0:
                    try:
                        os.remove(tmp_path)
                    except Exception:
                        pass

                with open(tmp_path, mode) as wf:
                    last_print = time.time()
                    while True:
                        chunk = r.read(1024 * 1024)
                        if not chunk:
                            break
                        wf.write(chunk)
                        total += len(chunk)
                        now = time.time()
                        if (now - last_print) >= 1.0:
                            sys.stdout.write(json.dumps({"downloading": os.path.basename(out_path), "mb": round(total / (1024 * 1024), 2)}, ensure_ascii=False) + "\n")
                            sys.stdout.flush()
                            last_print = now

            os.replace(tmp_path, out_path)
            if validate:
                _validate_sharegpt_json(out_path)
            return out_path
        except (URLError, TimeoutError, json.JSONDecodeError, ValueError, OSError) as e:
            last_err = e
            continue

    raise RuntimeError(f"Failed to download ShareGPT dataset: url={url} out={out_path} err={last_err}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=DEFAULT_SHAREGPT_URL)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--timeout-s", type=float, default=600.0)
    ap.add_argument("--retries", type=int, default=5)
    ap.add_argument("--no-validate", action="store_true")
    args = ap.parse_args()

    out = ensure_sharegpt_dataset(
        str(args.out),
        url=str(args.url),
        timeout_s=float(args.timeout_s),
        retries=int(args.retries),
        validate=not bool(args.no_validate),
    )
    print(json.dumps({"out": out}, ensure_ascii=False))


if __name__ == "__main__":
    main()
