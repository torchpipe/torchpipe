
# Copyright 2021-2026 NetEase.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#  http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from omniback.utils.system_path import system_include_dirs, system_library_dirs
import os
import sys
from ._cache_setting import get_cache_dir
import subprocess
import torch
import shutil

import logging
logger = logging.getLogger(__name__)  # type: ignore
cuda_version = int(torch.version.cuda.split('.')[0])

def is_system_exists_trt():
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

def can_use_trt_env():
    TENSORRT_INCLUDE = os.environ.get("TENSORRT_INCLUDE", None)
    TENSORRT_LIB = os.environ.get("TENSORRT_LIB", None)
    if TENSORRT_INCLUDE and TENSORRT_LIB:
        if not os.path.exists(TENSORRT_INCLUDE, "NvInfer.h"):
            raise RuntimeError(
                f"Wrong Env: TENSORRT_INCLUDE; can not find tensorrt header in dir {TENSORRT_INCLUDE}")
        if not os.path.exists(TENSORRT_LIB, "libnvinfer.so"):
            raise RuntimeError(
                f"Wrong Env: TENSORRT_LIB; can not find tensorrt libs in dir {TENSORRT_LIB}")
        return True
    return False
    
def get_trt_include_lib_dir():
    # from env
    TENSORRT_INCLUDE = os.environ.get("TENSORRT_INCLUDE", None)
    TENSORRT_LIB = os.environ.get("TENSORRT_LIB", None)
    if TENSORRT_INCLUDE and TENSORRT_LIB:
        if not os.path.exists(TENSORRT_INCLUDE, "NvInfer.h"):
            raise RuntimeError(
                f"Error Env TENSORRT_INCLUDE; can not find tensorrt header in dir {TENSORRT_INCLUDE}")
        if not os.path.exists(TENSORRT_LIB, "libnvinfer.so"):
            raise RuntimeError(
                f"Error Env TENSORRT_LIB; can not find tensorrt libs in dir {TENSORRT_LIB}")
        return TENSORRT_INCLUDE, TENSORRT_LIB
    # from cache
    TENSORRT_INCLUDE = TENSORRT_LIB = None
    cache_header = os.path.join(
        get_cache_dir(), f"tensorrt/tensorrt_cuda{cuda_version}/include/")
    cache_lib = os.path.join(
        get_cache_dir(), f"tensorrt/tensorrt_cuda{cuda_version}/lib/")
    possible_header_dirs = [cache_header]
    possible_lib_dirs = [cache_lib]
    for item in possible_header_dirs:
        if os.path.exists(os.path.join(item, "NvInfer.h")):
            TENSORRT_INCLUDE = item
            break
    for item in possible_lib_dirs:
        if os.path.exists(os.path.join(item, "libnvinfer.so")):
            TENSORRT_LIB = item
            break
    if TENSORRT_INCLUDE and TENSORRT_LIB:
        return TENSORRT_INCLUDE, TENSORRT_LIB

    return None, None

def get_sm():
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        sm_version = float(f"{props.major}.{props.minor}")
        return sm_version
    else:
        return 0
    
def cache_trt_dir():
    if cuda_version == 11:
        cuda_urls = ["https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/9.3.0/tensorrt-9.3.0.1.linux.x86_64-gnu.cuda-11.8.tar.gz"]
    elif cuda_version == 12:
        cuda_urls = ["https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/10.5.0/tars/TensorRT-10.5.0.18.Linux.x86_64-gnu.cuda-12.6.tar.gz",
                "https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/10.14.1/tars/TensorRT-10.14.1.48.Linux.x86_64-gnu.cuda-12.9.tar.gz"]
    elif cuda_version == 13:
        cuda_urls = ["https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/10.14.1/tars/TensorRT-10.14.1.48.Linux.x86_64-gnu.cuda-13.0.tar.gz"]
    
    trt_url = cuda_urls[0]
    TENSORRT_VERSION = trt_url.split(
        "machine-learning/tensorrt/")[1].split("/")[0]
    trt_file_name = trt_url.split('/')[-1]
    cache_dir = os.path.join(get_cache_dir(), "tensorrt")
    os.makedirs(cache_dir, exist_ok=True)
    # os.chdir(cache_dir)
    TRT_DIR = os.path.join(cache_dir, f"tensorrt_cuda{cuda_version}")

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
            logger.warning(
                f'You can set envs TENSORRT_INCLUDE and TENSORRT_LIB to skip downloading.')
            with open(tar_path+".cache", "wb") as f:
                with tqdm(
                    desc=f"Downloading {trt_file_name}",
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
        print(f"Extracting {trt_file_name} to {cache_dir} ...")
        import tarfile
        with tarfile.open(tar_path, "r:gz") as tar_ref:
            tar_ref.extractall(path=cache_dir)
            top_level_name = {m.split('/')[0]
                              for m in tar_ref.getnames() if m}.pop()
            if os.path.exists(TRT_DIR):
                shutil.rmtree(TRT_DIR)
            os.rename(os.path.join(cache_dir, top_level_name), TRT_DIR)
            os.remove(tar_path)
            
        print(f'saved to {TRT_DIR}')

    cache_header = os.path.join(TRT_DIR, "include/")
    cache_lib = os.path.join(TRT_DIR, "lib/")
    return cache_header, cache_lib


def _build_trt(csrc_dir):
    # python -m omniback.utils.build_lib --source-dirs csrc/tensorrt_torch/ --include-dirs=csrc/ --build-with-cuda --ldflags="-lnvinfer -lnvonnxparser  -lnvinfer_plugin" --name torchpipe_tensorrt

    if not is_system_exists_trt() and not can_use_trt_env():
        trt_inc, trt_lib = get_trt_include_lib_dir()
        if trt_inc is None:
            trt_inc, trt_lib = cache_trt_dir()
        if trt_inc is None:
            raise RuntimeError(
                "OpenCV not found. Please specify its location using the "
                "TENSORRT_INCLUDE and TENSORRT_LIB environment variables."
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
                f"--ldflags=-L{trt_lib} -lnvinfer -lnvonnxparser  -lnvinfer_plugin",
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
