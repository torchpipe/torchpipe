"""Encrypt model files with the TensorRT extension's compile-time key."""

from __future__ import annotations

import argparse
import ctypes
import os
from pathlib import Path

from omniback.utils import build_lib

from ..load_libs import _load_or_build_lib, get_whl_lib


def _resolve_tensorrt_lib_path() -> str:
    _load_or_build_lib("torchpipe_tensorrt")

    cache_lib = build_lib.get_cache_lib("torchpipe_tensorrt", "cuda", False)
    if os.path.exists(cache_lib):
        return cache_lib

    whl_lib = get_whl_lib(cache_lib)
    if whl_lib and os.path.exists(whl_lib):
        return whl_lib

    raise RuntimeError("torchpipe_tensorrt library is not available")


def _load_encrypt_symbol(lib_path: str):
    mode = getattr(ctypes, "RTLD_GLOBAL", 0)
    lib = ctypes.CDLL(lib_path, mode=mode)

    try:
        return lib.torchpipe_encrypt_file
    except AttributeError as exc:
        cache_lib = build_lib.get_cache_lib("torchpipe_tensorrt", "cuda", False)
        if os.path.exists(cache_lib) and os.path.samefile(lib_path, cache_lib):
            os.remove(cache_lib)
            raise RuntimeError(
                "Detected stale cached torchpipe_tensorrt library without "
                "`torchpipe_encrypt_file`. Removed cache; please rerun "
                "`python -m torchpipe.utils.encrypt` to rebuild it."
            ) from exc
        raise RuntimeError(
            f"`torchpipe_encrypt_file` is not exported by {lib_path}. "
            "Please rebuild torchpipe_tensorrt."
        ) from exc


def encrypt_file(input_path: str, output_path: str) -> None:
    lib_path = _resolve_tensorrt_lib_path()
    encrypt = _load_encrypt_symbol(lib_path)
    encrypt.argtypes = [
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.POINTER(ctypes.c_char),
        ctypes.c_size_t,
    ]
    encrypt.restype = ctypes.c_int

    error_message = ctypes.create_string_buffer(4096)
    status = encrypt(
        os.fsencode(input_path),
        os.fsencode(output_path),
        error_message,
        len(error_message),
    )
    if status != 0:
        detail = error_message.value.decode("utf-8", errors="replace").strip()
        raise RuntimeError(detail or f"encrypt failed with status {status}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Encrypt an ONNX/TRT file with torchpipe TensorRT runtime key.",
    )
    parser.add_argument("input", help="Input model path")
    parser.add_argument("output", help="Output encrypted file path")
    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"input file not found: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    encrypt_file(str(input_path), str(output_path))
    print(f"Encrypted file written to: {output_path}")


if __name__ == "__main__":
    main()
