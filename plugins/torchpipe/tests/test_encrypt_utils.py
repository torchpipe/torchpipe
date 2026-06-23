import hashlib
import logging
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from torchpipe.utils import _build_trt as build_trt
from torchpipe.utils import encrypt as encrypt_utils


def test_resolve_compile_time_key_hex_uses_sha256_of_env_secret(monkeypatch):
    monkeypatch.setenv("TORCHPIPE_TENSORRT_SECRET_KEY", "ab-cd 123")

    assert build_trt._resolve_compile_time_key_hex() == hashlib.sha256(
        b"ab-cd 123"
    ).hexdigest()


def test_build_trt_passes_compiled_key_hex_to_builder(monkeypatch):
    calls = {}

    monkeypatch.setattr(build_trt, "_resolve_compile_time_key_hex", lambda: "ab" * 32)
    monkeypatch.setattr(build_trt, "need_download_for_jit", lambda: False)
    monkeypatch.setattr(build_trt, "is_system_exists_trt", lambda: True)
    monkeypatch.setattr(build_trt, "can_use_trt_env", lambda: True)
    monkeypatch.setattr(build_trt, "get_trt_include_lib_dir", lambda: (None, None))
    monkeypatch.setattr(
        build_trt,
        "_build_tensorrt_extension",
        lambda csrc_dir, include_dirs, ldflags, extra_cflags: calls.update(
            {
                "csrc_dir": csrc_dir,
                "include_dirs": include_dirs,
                "ldflags": ldflags,
                "extra_cflags": extra_cflags,
            }
        ),
    )

    build_trt._build_trt("/tmp/fake-csrc", skip_download=True)

    assert calls["csrc_dir"] == "/tmp/fake-csrc"
    assert calls["include_dirs"] == []
    assert calls["ldflags"] == ["-lnvinfer", "-lnvonnxparser", "-lnvinfer_plugin"]
    assert calls["extra_cflags"] == ['-DTORCHPIPE_TENSORRT_KEY_HEX="{}"'.format("ab" * 32)]


def test_build_trt_warns_when_download_not_enabled(monkeypatch, caplog):
    monkeypatch.delenv("FORCE_DOWNLOAD_TENSORRT", raising=False)
    monkeypatch.setattr(build_trt, "_resolve_compile_time_key_hex", lambda: "ab" * 32)
    monkeypatch.setattr(build_trt, "need_download_for_jit", lambda: True)
    monkeypatch.setattr(
        build_trt,
        "_build_tensorrt_extension",
        lambda *args, **kwargs: pytest.fail("should not build without TensorRT"),
    )

    with caplog.at_level(logging.INFO):
        build_trt._build_trt("/tmp/fake-csrc", skip_download=True)

    assert "set FORCE_DOWNLOAD_TENSORRT=1 to download automatically" in caplog.text
    assert "downloading TensorRT into the cache" not in caplog.text


def test_build_trt_logs_download_attempt_when_force_download_enabled(monkeypatch, caplog):
    monkeypatch.setenv("FORCE_DOWNLOAD_TENSORRT", "1")
    monkeypatch.setattr(build_trt, "_resolve_compile_time_key_hex", lambda: "ab" * 32)
    monkeypatch.setattr(build_trt, "need_download_for_jit", lambda: True)
    monkeypatch.setattr(build_trt, "is_system_exists_trt", lambda: False)
    monkeypatch.setattr(build_trt, "can_use_trt_env", lambda: False)
    monkeypatch.setattr(build_trt, "get_trt_include_lib_dir", lambda: (None, None))
    monkeypatch.setattr(build_trt, "cache_trt_dir", lambda: (None, None))
    monkeypatch.setattr(
        build_trt,
        "_build_tensorrt_extension",
        lambda *args, **kwargs: pytest.fail("should not build when TensorRT remains unavailable"),
    )

    with caplog.at_level(logging.INFO):
        with pytest.raises(RuntimeError, match="download was attempted because FORCE_DOWNLOAD_TENSORRT=1"):
            build_trt._build_trt("/tmp/fake-csrc", skip_download=True)

    assert "downloading TensorRT into the cache" in caplog.text
    assert "Set FORCE_DOWNLOAD_TENSORRT=1 to download automatically" not in caplog.text


def test_encrypt_file_calls_exported_c_function(monkeypatch, tmp_path):
    calls = {}

    class FakeEncryptFunc:
        def __init__(self):
            self.argtypes = None
            self.restype = None

        def __call__(self, input_path, output_path, error_message, error_size):
            calls["input_path"] = input_path
            calls["output_path"] = output_path
            calls["error_size"] = error_size
            error_message.value = b""
            return 0

    class FakeLib:
        def __init__(self):
            self.torchpipe_encrypt_file = FakeEncryptFunc()

    monkeypatch.setattr(
        encrypt_utils,
        "_resolve_tensorrt_lib_path",
        lambda: "/tmp/fake_tensorrt.so",
    )
    monkeypatch.setattr(encrypt_utils.ctypes, "CDLL", lambda *args, **kwargs: FakeLib())

    input_path = tmp_path / "model.onnx"
    output_path = tmp_path / "model.onnx.encrypted"
    input_path.write_bytes(b"onnx-data")

    encrypt_utils.encrypt_file(str(input_path), str(output_path))

    assert calls["input_path"] == os.fsencode(input_path)
    assert calls["output_path"] == os.fsencode(output_path)
    assert calls["error_size"] == 4096


def test_encrypt_file_raises_runtime_error_from_native_message(monkeypatch, tmp_path):
    class FakeEncryptFunc:
        def __init__(self):
            self.argtypes = None
            self.restype = None

        def __call__(self, input_path, output_path, error_message, error_size):
            error_message.value = b"native failure"
            return -1

    class FakeLib:
        def __init__(self):
            self.torchpipe_encrypt_file = FakeEncryptFunc()

    monkeypatch.setattr(
        encrypt_utils,
        "_resolve_tensorrt_lib_path",
        lambda: "/tmp/fake_tensorrt.so",
    )
    monkeypatch.setattr(encrypt_utils.ctypes, "CDLL", lambda *args, **kwargs: FakeLib())

    input_path = tmp_path / "model.onnx"
    output_path = tmp_path / "model.onnx.encrypted"
    input_path.write_bytes(b"onnx-data")

    with pytest.raises(RuntimeError, match="native failure"):
        encrypt_utils.encrypt_file(str(input_path), str(output_path))


def test_load_encrypt_symbol_removes_stale_cached_lib(monkeypatch, tmp_path):
    stale_lib = tmp_path / "torchpipe_tensorrt.so"
    stale_lib.write_bytes(b"stub")

    class FakeLib:
        def __getattr__(self, name):
            raise AttributeError(name)

    monkeypatch.setattr(encrypt_utils.ctypes, "CDLL", lambda *args, **kwargs: FakeLib())
    monkeypatch.setattr(
        encrypt_utils.build_lib,
        "get_cache_lib",
        lambda name, device, no_torch: str(stale_lib),
    )

    with pytest.raises(RuntimeError, match="Removed cache; please rerun"):
        encrypt_utils._load_encrypt_symbol(str(stale_lib))

    assert not stale_lib.exists()


def test_main_prints_clear_output_message(monkeypatch, tmp_path, capsys):
    input_path = tmp_path / "model.onnx"
    output_path = tmp_path / "model.onnx.encrypted"
    input_path.write_bytes(b"onnx-data")

    monkeypatch.setattr(
        encrypt_utils,
        "encrypt_file",
        lambda input_file, output_file: None,
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "python -m torchpipe.utils.encrypt",
            str(input_path),
            str(output_path),
        ],
    )

    encrypt_utils.main()

    captured = capsys.readouterr()
    assert captured.out.strip() == f"Encrypted file written to: {output_path.resolve()}"


def test_cpp_encrypt_roundtrip_uses_sha256_key_derivation(tmp_path):
    gxx = shutil.which("g++")
    if gxx is None:
        pytest.skip("g++ not available")

    input_path = tmp_path / "model.bin"
    encrypted_path = tmp_path / "model.bin.encrypted"
    roundtrip_cpp = tmp_path / "roundtrip.cpp"
    roundtrip_bin = tmp_path / "roundtrip"
    input_bytes = bytes((index * 17 + 3) % 256 for index in range(380))
    input_path.write_bytes(input_bytes)

    csrc_dir = Path(__file__).resolve().parent.parent / "torchpipe" / "csrc" / "tensorrt_torch"
    roundtrip_cpp.write_text(
        f"""
#include <fstream>
#include <iostream>
#include <iterator>
#include <vector>

#include "encrypt.hpp"

int main() {{
  std::ifstream input_file("{input_path}", std::ios::binary);
  std::vector<unsigned char> original(
      (std::istreambuf_iterator<char>(input_file)),
      std::istreambuf_iterator<char>());

  torchpipe::encrypt_file_to_file("{input_path}", "{encrypted_path}");
  auto decrypted = torchpipe::decrypt_file("{encrypted_path}");
  if (decrypted != original) {{
    std::cerr << "roundtrip mismatch\\n";
    return 1;
  }}

  std::cout << "roundtrip ok\\n";
  return 0;
}}
""".strip()
        + "\n",
        encoding="utf-8",
    )

    compile_cmd = [
        gxx,
        "-std=c++17",
        '-DTORCHPIPE_TENSORRT_KEY_HEX="{}"'.format(
            hashlib.sha256(b"tp_roundtripcheck").hexdigest()
        ),
        f"-I{csrc_dir}",
        str(roundtrip_cpp),
        str(csrc_dir / "aes.cpp"),
        str(csrc_dir / "encrypt.cpp"),
        "-o",
        str(roundtrip_bin),
    ]
    subprocess.run(compile_cmd, check=True, capture_output=True, text=True)
    result = subprocess.run(
        [str(roundtrip_bin)],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.strip() == "roundtrip ok"
