
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
import subprocess
import os
import re
from ._cache_setting import get_cache_dir
import subprocess
from pathlib import Path

import logging
logger = logging.getLogger(__name__)  # type: ignore



def check_is_cxx11abi(lib_path):
    if not os.path.exists(lib_path):
        raise FileNotFoundError(f"Library not found: {lib_path}")

    try:
        # 使用 nm -D 获取动态符号表
        result = subprocess.run(['nm', '-D', lib_path], 
                                stdout=subprocess.PIPE, 
                                stderr=subprocess.PIPE, 
                                text=True)
        if result.returncode != 0:
            # fallback to readelf if nm fails (e.g., on stripped binaries)
            raise RuntimeError(f"nm failed: {result.stderr}")

        symbols = result.stdout

        # 检查是否存在 __cxx11（表示 C++11 ABI）
        if re.search(r'__cxx11', symbols):
            return True
        else:
            return False

    except Exception as e:
        result = subprocess.run(['readelf', '-Ws', lib_path],
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                                text=True)
        if ' __cxx11' in result.stdout or '_ZNSs' not in result.stdout:
            # 更准确：看是否有 std::__cxx11::string 符号
            if re.search(r'std::__cxx11::basic_string', result.stdout):
                return True
            elif re.search(r'std::basic_string', result.stdout) and not re.search(r'std::__cxx11::basic_string', result.stdout):
                return False
            else:
                raise RuntimeError(f"Unable to determine C++ ABI version")
        else:
            return False
    raise RuntimeError("Unable to determine C++ ABI version")

def is_system_exists_cv():
    exists_header = exists_lib = False
    for inc in system_include_dirs:
        if os.path.exists(os.path.join(inc, "opencv4/opencv2/core.hpp")):
            exists_header = True
            break
    for lib in system_library_dirs:
        cv_lib = os.path.join(lib, "libopencv_core.so")
        if os.path.exists(cv_lib):
            import omniback
            abiflag = (1==int(omniback.compiled_with_cxx11_abi()))
            if abiflag != check_is_cxx11abi(cv_lib):
                logger.warning(f"{cv_lib} and omniback is compiled with different cxx abi. Skip")
            else:
                exists_lib = True
            break
    return exists_lib and exists_header
def get_system_cv():
    exists_header = exists_lib = False
    for inc in system_include_dirs:
        if os.path.exists(os.path.join(inc, "opencv4/opencv2/core.hpp")):
            return os.path.join(inc, "opencv4/")
    raise RuntimeError('find no system opencv header')
    for lib in system_library_dirs:
        if os.path.exists(os.path.join(lib, "libopencv_core.so")):
            exists_lib = True
            break
    return exists_lib and exists_header

def can_use_cv_env():
    OPENCV_INCLUDE = os.environ.get("OPENCV_INCLUDE", None)
    OPENCV_LIB = os.environ.get("OPENCV_LIB", None)
    if OPENCV_INCLUDE and OPENCV_LIB:
        if not os.path.exists(OPENCV_INCLUDE, "opencv2/core.hpp"):
            raise RuntimeError(
                f"Wrong Env OPENCV_INCLUDE; can not find opencv header in dir {OPENCV_INCLUDE}")
        if not os.path.exists(OPENCV_LIB, "libopencv_core.so"):
            raise RuntimeError(
                f"Wrong Env OPENCV_LIB; can not find opencv libs in dir {OPENCV_LIB}")
        return True
    return False
    

def get_cv_include_lib_dir():
    # from env
    import omniback
    abiflag = int(omniback.compiled_with_cxx11_abi())
    OPENCV_INCLUDE = os.environ.get("OPENCV_INCLUDE", None)
    OPENCV_LIB = os.environ.get("OPENCV_LIB", None)
    if OPENCV_INCLUDE and OPENCV_LIB:
        if not os.path.exists(OPENCV_INCLUDE, "opencv2/core.hpp"):
            raise RuntimeError(
                f"can not find opencv header in dir {OPENCV_INCLUDE}")
        if not os.path.exists(OPENCV_LIB, "libopencv_core.so"):
            raise RuntimeError(f"can not find opencv libs in dir {OPENCV_LIB}")
        return OPENCV_INCLUDE, OPENCV_LIB
    # from cache
    OPENCV_INCLUDE = OPENCV_LIB = None
    cache_header = os.path.join(get_cache_dir(), f"opencv/abiflag{abiflag}/include/opencv4/")
    cache_lib = os.path.join(get_cache_dir(), "opencv/abiflag{abiflag}/lib/")
    possible_header_dirs = [cache_header]
    possible_lib_dirs = [cache_lib] +[os.path.join(get_cache_dir(), "opencv/abiflag{abiflag}/lib64/")]
    for item in possible_header_dirs:
        if os.path.exists(os.path.join(item, "opencv2/core.hpp")):
            OPENCV_INCLUDE = item
            break
    for item in possible_lib_dirs:
        if os.path.exists(os.path.join(item, "libopencv_core.so")):
            OPENCV_LIB = item
            break
    if OPENCV_INCLUDE and OPENCV_LIB:
        return OPENCV_INCLUDE, OPENCV_LIB

    return None, None

OPENCV_VERSION = os.environ.get("OPENCV_VERSION", "4.12.0")

def need_download_for_jit():
    if not is_system_exists_cv() and not can_use_cv_env():
        cv_inc, cv_lib = get_cv_include_lib_dir()
        if cv_inc is None:
            OPENCV_ZIP = f"opencv-{OPENCV_VERSION}.zip"

            cache_dir = Path(get_cache_dir()) / "opencv"

            OPENCV_DIR = cache_dir / f"opencv-{OPENCV_VERSION}"
            cmake_lists_path = OPENCV_DIR / "CMakeLists.txt"

            # Download and extract if not exists
            if not cmake_lists_path.exists():
                zip_path = cache_dir / OPENCV_ZIP
                if not zip_path.exists():
                    return True
    return False
            
def cache_cv_dir():
    import os
    import requests
    import zipfile
    import subprocess
    
    OPENCV_URL = f"https://codeload.github.com/opencv/opencv/zip/refs/tags/{OPENCV_VERSION}"
    OPENCV_ZIP = f"opencv-{OPENCV_VERSION}.zip"

    cache_dir = Path(get_cache_dir()) / "opencv"
    cache_dir.mkdir(parents=True, exist_ok=True)

    OPENCV_DIR = cache_dir / f"opencv-{OPENCV_VERSION}"
    cmake_lists_path = OPENCV_DIR / "CMakeLists.txt"

    # Download and extract if not exists
    if not cmake_lists_path.exists():
        zip_path = cache_dir / OPENCV_ZIP
        zip_path_cache = cache_dir / (OPENCV_ZIP+".cache")
        if not zip_path.exists():
            logger.warning(
                f"Downloading {OPENCV_URL} to {zip_path}")
            logger.warning(
                f'You may manully download it if it is too slow.')
            
            response = requests.get(OPENCV_URL, stream=True)
            response.raise_for_status()
            with open(zip_path_cache, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            os.rename(zip_path_cache, zip_path)

        # Extract OpenCV
        print(f"Extracting {OPENCV_ZIP} to {cache_dir}...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(cache_dir)

    # Prepare build directory
    build_dir = OPENCV_DIR / "build"
    build_dir.mkdir(exist_ok=True)
    
    logger.warning(f"Building OpenCV {OPENCV_VERSION}...")
    logger.info(f"Building in {build_dir}")
    logger.warning(
        f'You can set envs OPENCV_INCLUDE and OPENCV_LIB to skip the downloading/building steps.')
    import omniback
    abiflag = int(omniback.compiled_with_cxx11_abi())

    install_dir = cache_dir / f"abiflag{abiflag}"
    # CMake configuration
    cmake_args = [
        "cmake",
        f"-DCMAKE_CXX_FLAGS=-D_GLIBCXX_USE_CXX11_ABI={abiflag}",
        "-DCMAKE_BUILD_TYPE=Release",
        "-DBUILD_WITH_DEBUG_INFO=OFF",
        f"-DCMAKE_INSTALL_PREFIX={install_dir}",
        "-DINSTALL_C_EXAMPLES=OFF",
        "-DINSTALL_PYTHON_EXAMPLES=OFF",
        "-DENABLE_NEON=OFF",
        "-DBUILD_WEBP=OFF",
        "-DWITH_WEBP=OFF",
        "-DOPENCV_WEBP=OFF",
        "-DOPENCV_IO_ENABLE_WEBP=OFF",
        "-DHAVE_WEBP=OFF",
        "-DBUILD_ITT=OFF",
        "-DWITH_V4L=OFF",
        "-DWITH_QT=OFF",
        "-DWITH_OPENGL=OFF",
        "-DBUILD_opencv_dnn=OFF",
        "-DBUILD_dnn_plugins=OFF",
        "-DBUILD_opencv_java=OFF",
        "-DBUILD_opencv_python2=OFF",
        "-DBUILD_opencv_python3=OFF",
        "-DBUILD_NEW_PYTHON_SUPPORT=OFF",
        "-DBUILD_PYTHON_SUPPORT=OFF",
        "-DPYTHON_DEFAULT_EXECUTABLE=/usr/bin/python3",
        "-DBUILD_opencv_java_bindings_generator=OFF",
        "-DBUILD_opencv_python_bindings_generator=OFF",
        "-DBUILD_EXAMPLES=OFF",
        "-DWITH_OPENEXR=OFF",
        "-DWITH_JPEG=ON",
        "-DBUILD_JPEG=ON",
        "-DBUILD_JPEG_TURBO_DISABLE=OFF",
        "-DBUILD_DOCS=OFF",
        "-DBUILD_PERF_TESTS=OFF",
        "-DBUILD_TESTS=OFF",
        "-DWITH_PNG=OFF",
        "-DWITH_TIFF=OFF",
        "-DBUILD_opencv_apps=OFF",
        "-DBUILD_opencv_calib3d=OFF",
        "-DBUILD_opencv_contrib=OFF",
        "-DBUILD_opencv_features2d=OFF",
        "-DBUILD_opencv_flann=OFF",
        "-DBUILD_opencv_gapi=OFF",
        "-DWITH_CUDA=OFF",
        "-DWITH_CUDNN=OFF",
        "-DOPENCV_DNN_CUDA=OFF",
        "-DENABLE_FAST_MATH=1",
        "-DWITH_CUBLAS=0",
        "-DBUILD_opencv_gpu=OFF",
        "-DBUILD_opencv_ml=OFF",
        "-DBUILD_opencv_nonfree=OFF",
        "-DBUILD_opencv_objdetect=OFF",
        "-DBUILD_opencv_photo=OFF",
        "-DBUILD_opencv_stitching=OFF",
        "-DBUILD_opencv_superres=OFF",
        "-DBUILD_opencv_ts=OFF",
        "-DBUILD_opencv_highgui=OFF",
        "-DBUILD_highgui_plugins=OFF",
        "-DBUILD_opencv_video=OFF",
        "-DBUILD_opencv_videoio=OFF",
        "-DCMAKE_CXX_STANDARD=17",
        "-DCMAKE_CXX_STANDARD_REQUIRED=ON",
        "-DBUILD_videoio_plugins=OFF",
        "-DBUILD_opencv_videostab=OFF",
        "-DWITH_IPP=ON",
        "-DWITH_MKL=ON",
        "-DMKL_USE_TBB=ON",
        "-DWITH_TBB=ON",
        "-DBUILD_TBB=ON",
        "-DWITH_TURBOJPEG=ON",
        "-DWITH_LAPACK=ON",
        "-DWITH_BLAS=ON",
        str(OPENCV_DIR)  # Source directory
    ]

    # Build and install
    subprocess.run(cmake_args, cwd=build_dir, check=True, stdout=subprocess.DEVNULL,
                   stderr=subprocess.PIPE, 
                   text=True)
    subprocess.run(["make", "-j4"], cwd=build_dir, check=True)
    subprocess.run(["make", "install"], cwd=build_dir, check=True)

    # Return installation paths
    cache_header = install_dir / "include/opencv4/"
    cache_lib = install_dir / "lib/"
    if not cache_lib.exists():
        cache_lib = install_dir / "lib64/"
        
    return str(cache_header), str(cache_lib)


def _build_cv(csrc_dir, skip_download=True):
    # python -m omniback.utils.build_lib --no-torch --source-dirs csrc/mat_torch/ --include-dirs csrc/ /usr/local/include/opencv4/ --ldflags "-lopencv_core -lopencv_imgproc -lopencv_imgcodecs" --name torchpipe_opencv
    if skip_download and need_download_for_jit():
        FORCE_DOWNLOAD_OPENCV = os.environ.get("FORCE_DOWNLOAD_OPENCV", "0")
        if FORCE_DOWNLOAD_OPENCV == "0":
            logger.warning(
                f"Opencv not found. The system checked the following sources in order:\n"
                "  1. Environment variables OPENCV_INCLUDE and OPENCV_LIB (not set),\n"
                "  2. Standard system library paths (no valid OPENCV installation found).\n"
                "\n"
                "To proceed, please either:\n"
                "  - Set OPENCV_INCLUDE (e.g., /path/to/OPENCV/include/opencv4/) and OPENCV_LIB (e.g., /path/to/OPENCV/lib), or\n"
                f"  - Set FORCE_DOWNLOAD_OPENCV=1 to enable automatic download of OPENCV {OPENCV_VERSION}."
            )
            return

    if not is_system_exists_cv() and not can_use_cv_env():
        cv_inc, cv_lib = get_cv_include_lib_dir()
        if cv_inc is None:
            cv_inc, cv_lib = cache_cv_dir()
        if cv_inc is None:
            raise RuntimeError(
                "OpenCV not found. Please specify its location using the "
                "OPENCV_INCLUDE and OPENCV_LIB environment variables."
            )
        os.environ["LD_LIBRARY_PATH"] = f"{cv_lib}:" + \
            os.environ.get("LD_LIBRARY_PATH", "")

        subprocess.run(
            [
                sys.executable,
                "-m",
                "omniback.utils.build_lib",
                "--source-dirs",
                os.path.join(csrc_dir, "csrc/mat_torch/"),
                "--include-dirs",
                os.path.join(csrc_dir, "csrc/"),
                f"{cv_inc}",
                "--no-torch",
                f"--ldflags=-L{cv_lib} -lopencv_core -lopencv_imgproc -lopencv_imgcodecs",
                "--name",
                "torchpipe_opencv"
            ],
            check=True,
            env={**os.environ, "EXAMPLE_ENV": "1"},
        )
    else:
        cv_inc = get_system_cv()
        subprocess.run(
            [
                sys.executable,
                "-m",
                "omniback.utils.build_lib",
                "--source-dirs",
                os.path.join(csrc_dir, "csrc/mat_torch/"),
                "--include-dirs",
                os.path.join(csrc_dir, "csrc/"),
                f"{cv_inc}",
                "--no-torch",
                f"--ldflags= -lopencv_core -lopencv_imgproc -lopencv_imgcodecs",
                "--name",
                "torchpipe_opencv"
            ],
            check=True,
            env={**os.environ, "EXAMPLE_ENV": "1"},
        )
