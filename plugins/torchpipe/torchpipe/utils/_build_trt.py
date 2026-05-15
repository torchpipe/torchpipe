"""TensorRT build utilities for TorchPipe."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
from typing import Optional, Tuple

import torch

from omniback.utils.system_path import system_include_dirs, system_library_dirs
from ._cache_setting import get_cache_dir

logger = logging.getLogger(__name__)

_cuda_version = int(torch.version.cuda.split('.')[0])

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

    if skip_download and need_download_for_jit():
        logger.warning(
                "TensorRT not found. Checked:\n"
                "  1. Environment variables TENSORRT_INCLUDE and TENSORRT_LIB,\n"
                "  2. Standard system library paths\n"
                "  3. Cache directory\n"
                "\n"
                "Please either:\n"
                "  - Set TENSORRT_INCLUDE (e.g., /path/to/TensorRT/include) and TENSORRT_LIB (e.g., /path/to/TensorRT/lib), or\n"
                "  - Set FORCE_DOWNLOAD_TENSORRT=1 to download automatically, or\n"
                "  - Set TORCHPIPE_SKIP_TENSORRT=1 to skip TensorRT support entirely.\n"
                "You may also need to set LD_LIBRARY_PATH if not installed in a standard system path.\n"
            )
        FORCE_DOWNLOAD_TENSORRT = os.environ.get("FORCE_DOWNLOAD_TENSORRT", "0")
        if FORCE_DOWNLOAD_TENSORRT == "0":
            return

    # python -m omniback.utils.build_lib --source-dirs csrc/tensorrt_torch/ --include-dirs=csrc/ --build-with-cuda --ldflags="-lnvinfer -lnvonnxparser  -lnvinfer_plugin" --name torchpipe_tensorrt

    if not is_system_exists_trt() and not can_use_trt_env():
        trt_inc, trt_lib = get_trt_include_lib_dir()
        if trt_inc is None:
            trt_inc, trt_lib = cache_trt_dir()
        if trt_inc is None:
            raise RuntimeError(
                "TensorRT not found. Please specify its location using the "
                "TENSORRT_INCLUDE and TENSORRT_LIB environment variables, "
                "or set FORCE_DOWNLOAD_TENSORRT=1 to automatically download."
            )
        os.environ["LD_LIBRARY_PATH"] = f"{trt_lib}:" + \
            os.environ.get("LD_LIBRARY_PATH", "")

        subprocess.run(
            [
                sys.executable,
                "-m",
                "omniback.utils.build_lib",
                "--source-dirs",
                os.path.join(csrc_dir, "csrc/tensorrt_torch/"),
                "--include-dirs",
                os.path.join(csrc_dir, "csrc/"),
                f"{trt_inc}",
                "--build-with-cuda",
                # f"--ldflags=-L{trt_lib} -lnvinfer -lnvonnxparser  -lnvinfer_plugin",
                f"--ldflags=-L{trt_lib} -Wl,-rpath,{trt_lib} -lnvinfer -lnvonnxparser -lnvinfer_plugin",
                "--name",
                "torchpipe_tensorrt"
            ],
            check=True,
            env={**os.environ, "EXAMPLE_ENV": "1"},
        )
    else:
        trt_inc, trt_lib = get_trt_include_lib_dir()
        if trt_inc and trt_lib:
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "omniback.utils.build_lib",
                    "--source-dirs",
                    os.path.join(csrc_dir, "csrc/tensorrt_torch/"),
                    "--include-dirs",
                    os.path.join(csrc_dir, "csrc/"),
                    trt_inc,
                    "--build-with-cuda",
                    f"--ldflags=-L{trt_lib} -Wl,-rpath,{trt_lib} -lnvinfer -lnvonnxparser -lnvinfer_plugin",
                    "--name",
                    "torchpipe_tensorrt"
                ],
                check=True,
                env={**os.environ, "EXAMPLE_ENV": "1"},
            )
        else:
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "omniback.utils.build_lib",
                    "--source-dirs",
                    os.path.join(csrc_dir, "csrc/tensorrt_torch/"),
                    "--include-dirs",
                    os.path.join(csrc_dir, "csrc/"),
                    "--build-with-cuda",
                    f"--ldflags=-lnvinfer -lnvonnxparser -lnvinfer_plugin",
                    "--name",
                    "torchpipe_tensorrt"
                ],
                check=True,
                env={**os.environ, "EXAMPLE_ENV": "1"},
            )
