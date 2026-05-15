from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--opset", type=int, default=21)
    parser.add_argument("--dtype", type=str, default="fp16")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--slim", action="store_true", default=True)
    args = parser.parse_args(argv)

    exe = shutil.which("optimum-cli")
    if exe is None:
        raise RuntimeError("optimum-cli not found in PATH")

    if args.out_dir:
        out_dir = os.path.abspath(args.out_dir)
    else:
        from ..cache import cached_model_dir

        out_dir = cached_model_dir(args.model_id, dtype=args.dtype)
    os.makedirs(out_dir, exist_ok=True)

    cmd = [
        exe,
        "export",
        "onnx",
        "--model",
        args.model_id,
        out_dir,
        f"--opset={int(args.opset)}",
        f"--dtype={args.dtype}",
        f"--device={args.device}",
    ]
    if args.slim:
        cmd.append("--slim")

    p = subprocess.run(cmd, check=False)
    if p.returncode != 0:
        raise SystemExit(int(p.returncode))


if __name__ == "__main__":
    main()
