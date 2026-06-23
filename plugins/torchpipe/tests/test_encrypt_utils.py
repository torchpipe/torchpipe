import os
import shutil
import subprocess
from pathlib import Path

import pytest

from torchpipe.utils import _build_trt as build_trt
from torchpipe.utils import encrypt as encrypt_utils


def test_prepare_secret_key_include_dir_uses_temp_dir_and_writes_header(monkeypatch):
    monkeypatch.setenv("TORCHPIPE_TENSORRT_SECRET_KEY", "ab-cd 123")

    include_dir, temp_dir = build_trt._prepare_secret_key_include_dir()
    try:
        include_path = Path(include_dir)
        assert include_path.parent == Path(temp_dir)

        header_path = include_path / "torchpipe_tensorrt_secret_key.hpp"
        assert header_path.exists()

        content = header_path.read_text(encoding="ascii")
        assert '#define SECRET_KEY tp_ab_cd_123' in content
    finally:
        build_trt.shutil.rmtree(temp_dir, ignore_errors=True)


def test_build_trt_cleans_up_temp_secret_dir_on_failure(monkeypatch, tmp_path):
    temp_dir = tmp_path / "secret-key-dir"
    include_dir = temp_dir / "include"
    include_dir.mkdir(parents=True)
    (include_dir / "torchpipe_tensorrt_secret_key.hpp").write_text(
        "#define SECRET_KEY tp_test\n",
        encoding="ascii",
    )

    monkeypatch.setattr(
        build_trt,
        "_prepare_secret_key_include_dir",
        lambda: (str(include_dir), str(temp_dir)),
    )
    monkeypatch.setattr(build_trt, "need_download_for_jit", lambda: False)
    monkeypatch.setattr(build_trt, "is_system_exists_trt", lambda: True)
    monkeypatch.setattr(build_trt, "can_use_trt_env", lambda: True)
    monkeypatch.setattr(build_trt, "get_trt_include_lib_dir", lambda: (None, None))

    def fake_run(*args, **kwargs):
        raise subprocess.CalledProcessError(returncode=1, cmd=args[0])

    monkeypatch.setattr(build_trt.subprocess, "run", fake_run)

    with pytest.raises(subprocess.CalledProcessError):
        build_trt._build_trt("/tmp/fake-csrc", skip_download=True)

    assert not temp_dir.exists()


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

    csrc_dir = Path(
        "/mnt/data2/zhangshiyang/workspace/torchpipe/plugins/torchpipe/torchpipe/csrc/tensorrt_torch"
    )
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
        "-DSECRET_KEY=tp_roundtripcheck",
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
