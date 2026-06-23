"""TensorRT build utilities for TorchPipe."""

from __future__ import annotations

import hashlib
import logging
import os
import secrets
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional, Tuple

import torch

from omniback.utils import build_lib as omniback_build_lib
from omniback.utils.system_path import system_include_dirs, system_library_dirs
from ._cache_setting import get_cache_dir

logger = logging.getLogger(__name__)

_cuda_version = int(torch.version.cuda.split('.')[0])


def _resolve_compile_time_key_hex() -> str:
    """Return a deterministic 32-byte key for TensorRT encryption."""
    secret_key = os.environ.get("TORCHPIPE_TENSORRT_SECRET_KEY")
    if not secret_key:
        return secrets.token_hex(32)
    return hashlib.sha256(secret_key.encode("utf-8")).hexdigest()


def _build_tensorrt_extension(
        csrc_dir: str,
        include_dirs: list[str],
        ldflags: list[str],
        extra_cflags: list[str]) -> None:
    """Build the TensorRT extension without emitting a temporary header file."""
    os.environ["TVM_FFI_DISABLE_TORCH_C_DLPACK"] = "1"
    from tvm_ffi.cpp.extension import build
    from tvm_ffi.utils.lockfile import FileLock
    from torch.utils.cpp_extension import CUDA_HOME, library_paths
    import omniback as om
    from omniback import get_include_dirs

    abiflag = "1" if torch.compiled_with_cxx11_abi() else "0"
    libname = omniback_build_lib.get_cache_name("torchpipe_tensorrt", "cuda", False, abiflag)
    tmp_libname = libname + ".tmp"
    suffix = ".dll" if omniback_build_lib.IS_WINDOWS else ".so"
    output_dir = Path(omniback_build_lib.get_cache_dir()).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    final_path = output_dir / f"{libname}{suffix}"

    with FileLock(str(output_dir / f"{libname}.lock")):
        if final_path.exists():
            return

        source_dirs = [Path(csrc_dir).expanduser() / "csrc" / "tensorrt_torch"]
        source_path = omniback_build_lib.get_cpp_source([str(path) for path in source_dirs])

        include_paths = [str((Path(csrc_dir).expanduser() / "csrc").resolve())]
        include_paths.extend(str(Path(path).expanduser().resolve()) for path in include_dirs)

        cflags = [
            f"-D_GLIBCXX_USE_CXX11_ABI={abiflag}",
            "-DBUILD_WITH_CUDA",
            *extra_cflags,
        ]
        link_flags = list(ldflags)

        omniback_build_lib._check_pkg_resources()
        include_paths.extend(omniback_build_lib.get_torch_include_paths(True))

        for lib_dir in library_paths():
            if omniback_build_lib.IS_WINDOWS:
                link_flags.append(f"/LIBPATH:{lib_dir}")
            else:
                link_flags.extend(["-L", str(lib_dir)])

        if CUDA_HOME is None:
            if hasattr(torch, "version") and getattr(torch.version, "cuda", None):
                torch_dir = os.path.dirname(torch.__file__)
                if os.path.exists(os.path.join(torch_dir, "lib")):
                    CUDA_HOME = torch_dir
                    os.environ["CUDA_HOME"] = CUDA_HOME
                    logger.warning(
                        "CUDA_HOME not found. Falling back to PyTorch's internal CUDA path: %s",
                        CUDA_HOME,
                    )
        if CUDA_HOME is None:
            logger.error("CUDA_HOME not found")
        else:
            cuda_lib_dir = os.path.join(
                CUDA_HOME,
                "lib64" if os.path.exists(os.path.join(CUDA_HOME, "lib64")) else "lib",
            )
            if omniback_build_lib.IS_WINDOWS:
                link_flags.append(f"/LIBPATH:{cuda_lib_dir}")
            else:
                link_flags.extend(["-L", str(cuda_lib_dir)])

        if omniback_build_lib.IS_WINDOWS:
            link_flags.extend(["c10.lib", "torch.lib", "torch_cpu.lib", "torch_cuda.lib", "c10_cuda.lib"])
        else:
            link_flags.extend(["-lc10", "-ltorch", "-ltorch_cpu", "-ltorch_cuda", "-lc10_cuda"])

        om_lib = om.libinfo.find_libomniback()
        link_flags.append(f"-L{os.path.dirname(om_lib)}")
        om_lib_name = os.path.splitext(os.path.basename(om_lib))[0].strip("lib")
        if om_lib_name.startswith("lib"):
            om_lib_name = om_lib_name[3:]
        link_flags.append(f"-l{om_lib_name}")

        include_paths += get_include_dirs()
        include_paths = omniback_build_lib.unique_paths([str(path) for path in include_paths])

        with tempfile.TemporaryDirectory(prefix="omniback-torch") as build_dir:
            result_lib = build(
                name=tmp_libname,
                cpp_files=[str(path) for path in source_path],
                extra_cflags=cflags,
                extra_ldflags=link_flags,
                extra_include_paths=include_paths,
                build_directory=build_dir,
            )
            shutil.move(
                str(result_lib),
                str(final_path),
            )
            print(f"saved to {final_path}")

def is_system_exists_trt() -> bool:
    """Check if TensorRT is available in system paths."""
    exists_header = exists_lib = False
    for inc in system_include_dirs:
        if os.path.exists(os.path.join(inc, "NvInfer.h")):
            exists_header = True
            break
    for lib in system_library_dirs:
        if os.path.exists(os.path.join(lib, "libnvinfer.so")):
            exists_lib = True
            break
    return exists_lib and exists_header

def can_use_trt_env() -> bool:
    """Check if TensorRT paths are set in environment variables."""
    tensorrt_include = os.environ.get("TENSORRT_INCLUDE")
    tensorrt_lib = os.environ.get("TENSORRT_LIB")
    if tensorrt_include and tensorrt_lib:
        if not os.path.exists(tensorrt_include):
            logger.warning("TENSORRT_INCLUDE path does not exist: %s", tensorrt_include)
            return False
        if not os.path.exists(tensorrt_lib):
            logger.warning("TENSORRT_LIB path does not exist: %s", tensorrt_lib)
            return False
        if not os.path.exists(os.path.join(tensorrt_include, "NvInfer.h")):
            logger.warning("TENSORRT_INCLUDE invalid: NvInfer.h not found in %s", tensorrt_include)
            return False
        if not os.path.exists(os.path.join(tensorrt_lib, "libnvinfer.so")):
            logger.warning("TENSORRT_LIB invalid: libnvinfer.so not found in %s", tensorrt_lib)
            return False
        return True
    return False

def get_trt_include_lib_dir() -> Tuple[Optional[str], Optional[str]]:
    """Get TensorRT include and library directories.

    Returns:
        Tuple of (include_dir, lib_dir) or (None, None) if not found
    """
    # Check environment variables first
    tensorrt_include = os.environ.get("TENSORRT_INCLUDE")
    tensorrt_lib = os.environ.get("TENSORRT_LIB")
    if tensorrt_include and tensorrt_lib:
        if os.path.exists(tensorrt_include) and os.path.exists(tensorrt_lib):
            if os.path.exists(os.path.join(tensorrt_include, "NvInfer.h")):
                if os.path.exists(os.path.join(tensorrt_lib, "libnvinfer.so")):
                    return tensorrt_include, tensorrt_lib
                else:
                    logger.debug("libnvinfer.so not found in %s", tensorrt_lib)
            else:
                logger.debug("NvInfer.h not found in %s", tensorrt_include)
        else:
            if not os.path.exists(tensorrt_include):
                logger.debug("TENSORRT_INCLUDE path does not exist: %s", tensorrt_include)
            if not os.path.exists(tensorrt_lib):
                logger.debug("TENSORRT_LIB path does not exist: %s", tensorrt_lib)

    # Check cache directory
    cache_header = os.path.join(
        get_cache_dir(), f"tensorrt/tensorrt_cuda{_cuda_version}/include/")
    cache_lib = os.path.join(
        get_cache_dir(), f"tensorrt/tensorrt_cuda{_cuda_version}/lib/")

    if os.path.exists(cache_header) and os.path.exists(cache_lib):
        if os.path.exists(os.path.join(cache_header, "NvInfer.h")):
            if os.path.exists(os.path.join(cache_lib, "libnvinfer.so")):
                return cache_header, cache_lib

    return None, None

def get_sm() -> float:
    """Get GPU compute capability (SM version)."""
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        return float(f"{props.major}.{props.minor}")
    return 0.0

def get_trt_url() -> str:
    """Get TensorRT download URL for current CUDA version."""
    if _cuda_version == 11:
        return "https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/9.3.0/tensorrt-9.3.0.1.linux.x86_64-gnu.cuda-11.8.tar.gz"
    elif _cuda_version == 12:
        sm = get_sm()
        if sm >= 12.0:
            return "https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/10.14.1/tars/TensorRT-10.14.1.48.Linux.x86_64-gnu.cuda-12.9.tar.gz"
        elif sm <= 6.1:
            return "https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/9.3.0/tensorrt-9.3.0.1.linux.x86_64-gnu.cuda-12.2.tar.gz"
        else:
            return "https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/10.5.0/tars/TensorRT-10.5.0.18.Linux.x86_64-gnu.cuda-12.6.tar.gz"
    elif _cuda_version == 13:
        return "https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/10.14.1/tars/TensorRT-10.14.1.48.Linux.x86_64-gnu.cuda-13.0.tar.gz"
    else:
        raise RuntimeError(f"Unsupported CUDA version: {_cuda_version}")

def need_download_trt_for_cache():
    trt_url = get_trt_url()
    trt_file_name = trt_url.split('/')[-1]
    cache_dir = os.path.join(get_cache_dir(), "tensorrt")
    TRT_DIR = os.path.join(cache_dir, f"tensorrt_cuda{_cuda_version}")

    core_files = [
        "lib/libnvinfer.so",
        "lib/libnvonnxparser.so",
        "include/NvInfer.h",
        "lib/libnvinfer_plugin.so",
        "include/NvInferPlugin.h",
    ]
    if not all(os.path.exists(os.path.join(TRT_DIR, f)) for f in core_files):
        tar_path = os.path.join(cache_dir, trt_file_name)
        if not os.path.exists(tar_path):
            return True
    return False

def cache_trt_dir():
    trt_url = get_trt_url()
    TENSORRT_VERSION = trt_url.split(
        "machine-learning/tensorrt/")[1].split("/")[0]
    trt_file_name = trt_url.split('/')[-1]
    cache_dir = os.path.join(get_cache_dir(), "tensorrt")
    os.makedirs(cache_dir, exist_ok=True)
    # os.chdir(cache_dir)
    TRT_DIR = os.path.join(cache_dir, f"tensorrt_cuda{_cuda_version}")

    cache_header = os.path.join(TRT_DIR, "include/")
    cache_lib = os.path.join(TRT_DIR, "lib/")

    core_files = [
        "lib/libnvinfer.so",
        "lib/libnvonnxparser.so",
        "include/NvInfer.h",
        "lib/libnvinfer_plugin.so",
        "include/NvInferPlugin.h",
    ]
    if not all(os.path.exists(os.path.join(TRT_DIR, f)) for f in core_files):
        tar_path = os.path.join(cache_dir, trt_file_name)
        if not os.path.exists(tar_path):
            import requests
            from tqdm import tqdm

            response = requests.get(trt_url, stream=True)
            response.raise_for_status()  # 确保请求成功

            # 获取文件总大小（注意：有些服务器可能不提供 Content-Length）
            total_size = int(response.headers.get('content-length', 0))
            logger.warning(f"\nDownloading {trt_file_name}. You may need to set LD_LIBRARY_PATH={cache_lib}:$LD_LIBRARY_PATH after installation.")
            with open(tar_path+".cache", "wb") as f:
                with tqdm(
                    desc=f"",
                    total=total_size,
                    unit='B',
                    unit_scale=True,
                    unit_divisor=1024,
                ) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:  # 过滤掉 keep-alive 空块
                            f.write(chunk)
                            pbar.update(len(chunk))
            os.rename(tar_path+".cache", tar_path)
        print(f"Extracting {trt_file_name} to {cache_dir}")
        import tarfile
        with tarfile.open(tar_path, "r:gz") as tar_ref:
            tar_ref.extractall(path=cache_dir)
            top_level_name = {m.split('/')[0]
                              for m in tar_ref.getnames() if m}.pop()
            if os.path.exists(TRT_DIR):
                shutil.rmtree(TRT_DIR)
            os.rename(os.path.join(cache_dir, top_level_name), TRT_DIR)
            os.remove(tar_path)

        logger.warning(f'saved to {TRT_DIR}/')

    return cache_header, cache_lib


def need_download_for_jit():
    if not is_system_exists_trt() and not can_use_trt_env():
        trt_inc, trt_lib = get_trt_include_lib_dir()
        if trt_inc is None:
            return need_download_trt_for_cache()
    return False

def _build_trt(csrc_dir, skip_download=True):
    # Check if we should skip TensorRT entirely
    if os.environ.get("TORCHPIPE_SKIP_TENSORRT", "0") == "1":
        logger.info("TORCHPIPE_SKIP_TENSORRT=1, skipping TensorRT build")
        return

    key_hex = _resolve_compile_time_key_hex()
    extra_cflags = [f"-DTORCHPIPE_TENSORRT_KEY_HEX={key_hex}"]
    force_download_tensorrt = os.environ.get("FORCE_DOWNLOAD_TENSORRT", "0")
    if skip_download and need_download_for_jit():
        if force_download_tensorrt == "0":
            logger.warning(
                "TensorRT not found in environment variables, system library paths, or cache.\n"
                "Set TENSORRT_INCLUDE/TENSORRT_LIB, set FORCE_DOWNLOAD_TENSORRT=1 to download automatically,\n"
                "or set TORCHPIPE_SKIP_TENSORRT=1 to skip TensorRT support."
            )
            return
        logger.info(
            "TensorRT not found locally; FORCE_DOWNLOAD_TENSORRT=1, downloading TensorRT into the cache."
        )

    if not is_system_exists_trt() and not can_use_trt_env():
        trt_inc, trt_lib = get_trt_include_lib_dir()
        if trt_inc is None:
            trt_inc, trt_lib = cache_trt_dir()
        if trt_inc is None:
            if force_download_tensorrt != "0":
                raise RuntimeError(
                    "TensorRT download was attempted because FORCE_DOWNLOAD_TENSORRT=1, "
                    "but TensorRT is still unavailable. Set TENSORRT_INCLUDE/TENSORRT_LIB "
                    "manually or inspect the download/cache state."
                )
            raise RuntimeError(
                "TensorRT not found. Please specify its location using the "
                "TENSORRT_INCLUDE and TENSORRT_LIB environment variables, "
                "or set FORCE_DOWNLOAD_TENSORRT=1 to automatically download."
            )
        os.environ["LD_LIBRARY_PATH"] = f"{trt_lib}:" + \
            os.environ.get("LD_LIBRARY_PATH", "")
        _build_tensorrt_extension(
            csrc_dir,
            include_dirs=[trt_inc],
            ldflags=[f"-L{trt_lib}", f"-Wl,-rpath,{trt_lib}", "-lnvinfer", "-lnvonnxparser", "-lnvinfer_plugin"],
            extra_cflags=extra_cflags,
        )
    else:
        trt_inc, trt_lib = get_trt_include_lib_dir()
        if trt_inc and trt_lib:
            _build_tensorrt_extension(
                csrc_dir,
                include_dirs=[trt_inc],
                ldflags=[f"-L{trt_lib}", f"-Wl,-rpath,{trt_lib}", "-lnvinfer", "-lnvonnxparser", "-lnvinfer_plugin"],
                extra_cflags=extra_cflags,
            )
        else:
            _build_tensorrt_extension(
                csrc_dir,
                include_dirs=[],
                ldflags=["-lnvinfer", "-lnvonnxparser", "-lnvinfer_plugin"],
                extra_cflags=extra_cflags,
            )
