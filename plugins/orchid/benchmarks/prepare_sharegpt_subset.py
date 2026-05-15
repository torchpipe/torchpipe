import argparse
import json
import os
import time
from typing import Any
import sys

from orchid.paths import benchmark_dataset


def _ensure_dir(path: str) -> None:
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)


def _now_tag() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())


def _normalize_entry(e: dict[str, Any]) -> dict[str, Any] | None:
    conv = e.get("conversations")
    if isinstance(conv, list) and len(conv) >= 2:
        a = conv[0]
        b = conv[1]
        if isinstance(a, dict) and isinstance(b, dict) and "value" in a and "value" in b:
            return {
                "conversations": [
                    {"from": "human", "value": str(a["value"])},
                    {"from": "gpt", "value": str(b["value"])},
                ]
            }

    conv2 = e.get("conversation")
    if isinstance(conv2, list) and len(conv2) >= 2:
        out_conv = []
        for turn in conv2:
            if not isinstance(turn, dict):
                continue
            for k, v in turn.items():
                if k == "assistant":
                    out_conv.append({"from": "gpt", "value": str(v)})
                elif k == "user":
                    out_conv.append({"from": "human", "value": str(v)})
        if len(out_conv) >= 2:
            return {"conversations": out_conv[:2]}
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-id", default="shareAI/ShareGPT-Chinese-English-90k")
    ap.add_argument("--split", default="train")
    ap.add_argument("--num-samples", type=int, default=2000)
    ap.add_argument("--out", default=benchmark_dataset("sharegpt_subset.json"))
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    out = os.path.abspath(str(args.out))
    if str(args.tag).strip():
        base, ext = os.path.splitext(out)
        out = f"{base}.{str(args.tag).strip()}{ext or '.json'}"
    else:
        base, ext = os.path.splitext(out)
        out = f"{base}.{_now_tag()}{ext or '.json'}"

    _ensure_dir(out)

    import datasets

    ds = datasets.load_dataset(str(args.dataset_id), split=str(args.split), streaming=True)
    data: list[dict[str, Any]] = []
    want = int(args.num_samples)
    for x in ds:
        if len(data) >= want:
            break
        if not isinstance(x, dict):
            continue
        y = _normalize_entry(x)
        if y is None:
            continue
        data.append(y)

    with open(out, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)

    sys_out = {"out": out, "count": len(data), "dataset_id": str(args.dataset_id), "split": str(args.split)}
    print(json.dumps(sys_out, ensure_ascii=False))


if __name__ == "__main__":
    main()
