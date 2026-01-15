import subprocess
from pathlib import Path
from omniback.utils import build_lib
import ctypes, os, sys
import logging
logger = logging.getLogger(__name__)  # type: ignore

csrc_dir = os.path.dirname(__file__)
def load_whl_lib(path_of_cache):
    p = os.path.join(os.path.dirname(__file__),
                     os.path.basename(path_of_cache))
    if os.path.exists(p):
        ctypes.CDLL(p, mode=ctypes.RTLD_GLOBAL)
        return True
    return False

def _load_lib_with_torch_cuda(name):
    local_lib = build_lib.get_cache_lib(
        name, "cuda", False)
    if load_whl_lib(local_lib):
            return True
    if os.path.exists(local_lib):
        ctypes.CDLL(local_lib, mode=ctypes.RTLD_GLOBAL)
        return True
    return False

def _load_lib(name):
    if name == "torchpipe_opencv":
        torchpipe_opencv = build_lib.get_cache_lib(
            "torchpipe_opencv", "", True)
        if load_whl_lib(torchpipe_opencv):
            return True
        if os.path.exists(torchpipe_opencv):
            try:
                ctypes.CDLL(torchpipe_opencv, mode=ctypes.RTLD_GLOBAL)
            except OSError:
                cache_lib = os.path.join(get_cache_dir(), "opencv/lib")
                OPENCV_LIB = os.environ.get("OPENCV_LIB", cache_lib)
                core = Path(OPENCV_LIB)/"libopencv_core.so"
                imgproc = Path(OPENCV_LIB)/"libopencv_imgproc.so"
                imgcodecs = Path(OPENCV_LIB)/"libopencv_imgcodecs.so"
                
                ctypes.CDLL(core.resolve(), mode=ctypes.RTLD_GLOBAL)
                ctypes.CDLL(imgproc.resolve(), mode=ctypes.RTLD_GLOBAL)
                ctypes.CDLL(imgcodecs.resolve(), mode=ctypes.RTLD_GLOBAL)
                    
                ctypes.CDLL(torchpipe_opencv, mode=ctypes.RTLD_GLOBAL)
            return True
    else:
        return _load_lib_with_torch_cuda(name)
    return False


def get_cache_dir():
    cache= str(Path(os.environ.get("OMNIBACK_CACHE_DIR",
                                   "~/.cache/omniback/")).expanduser())
    return os.path.join(cache, "torchpipe/")

def get_cv_include_lib_dir():
    # from env
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
    cache_header = os.path.join(get_cache_dir(), "opencv/include/opencv4/")
    cache_lib = os.path.join(get_cache_dir(), "opencv/lib/")
    possible_header_dirs = [cache_header, "/usr/local/include/opencv4/"]
    possible_lib_dirs = [cache_lib, "/usr/local/lib/"]
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

def cache_cv_dir():
    OPENCV_VERSION = "4.5.4"
    OPENCV_URL = f"https://codeload.github.com/opencv/opencv/zip/refs/tags/{OPENCV_VERSION}"
    OPENCV_ZIP = f"opencv-{OPENCV_VERSION}.zip"
    cache_dir = os.path.join(get_cache_dir(), "opencv")
    os.makedirs(cache_dir, exist_ok=True)
    os.chdir(cache_dir)
            
    OPENCV_DIR = os.path.join(cache_dir, f"opencv-{OPENCV_VERSION}")
    if not os.path.exists(os.path.join(OPENCV_DIR, "CMakeLists.txt")):
        if not os.path.exists(os.path.join(cache_dir, OPENCV_ZIP)):
            import requests
            response = requests.get(OPENCV_URL, stream=True)
            with open(OPENCV_ZIP, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

        # Extract OpenCV
        print(f"Extracting {OPENCV_ZIP}...")
        import zipfile
        with zipfile.ZipFile(OPENCV_ZIP, "r") as zip_ref:
            zip_ref.extractall()
    os.chdir(OPENCV_DIR)
    print(f"build in {OPENCV_DIR}")
    import omniback
    abi_flag = int(omniback.compiled_with_cxx11_abi())
    
    os.makedirs("build", exist_ok=True)
    os.chdir("build")
    subprocess.run([
        "cmake",
        "-DCMAKE_CXX_FLAGS=-D_GLIBCXX_USE_CXX11_ABI=" + str(abi_flag),
        "-D", "CMAKE_BUILD_TYPE=Release",
        "-D", "BUILD_WITH_DEBUG_INFO=OFF",
        "-D", f"CMAKE_INSTALL_PREFIX={cache_dir}",
        "-D", "INSTALL_C_EXAMPLES=OFF",
        "-D", "INSTALL_PYTHON_EXAMPLES=OFF",
        "-D", "ENABLE_NEON=OFF",
        "-D", "BUILD_WEBP=OFF",
        "-D", "WITH_WEBP=OFF",
        "-D", "OPENCV_WEBP=OFF",
        "-D", "OPENCV_IO_ENABLE_WEBP=OFF",
        "-D", "HAVE_WEBP=OFF",
        "-D", "BUILD_ITT=OFF",
        "-D", "WITH_V4L=OFF",
        "-D", "WITH_QT=OFF",
        "-D", "WITH_OPENGL=OFF",
        "-D", "BUILD_opencv_dnn=OFF",
        "-D", "BUILD_opencv_java=OFF",
        "-D", "BUILD_opencv_python2=OFF",
        "-D", "BUILD_opencv_python3=OFF",
        "-D", "BUILD_NEW_PYTHON_SUPPORT=OFF",
        "-D", "BUILD_PYTHON_SUPPORT=OFF",
        "-D", "PYTHON_DEFAULT_EXECUTABLE=/usr/bin/python3",
        "-D", "BUILD_opencv_java_bindings_generator=OFF",
        "-D", "BUILD_opencv_python_bindings_generator=OFF",
        "-D", "BUILD_EXAMPLES=OFF",
        "-D", "WITH_OPENEXR=OFF",
        "-D", "WITH_JPEG=ON",
        "-D", "BUILD_JPEG=ON",
        "-D", "BUILD_JPEG_TURBO_DISABLE=OFF",
        "-D", "BUILD_DOCS=OFF",
        "-D", "BUILD_PERF_TESTS=OFF",
        "-D", "BUILD_TESTS=OFF",
        "-D", "WITH_PNG=OFF",      # 如果不需要 PNG
        "-D", "WITH_TIFF=OFF",     # 如果不需要 TIFF
        "-D", "BUILD_opencv_apps=OFF",
        "-D", "BUILD_opencv_calib3d=OFF",
        "-D", "BUILD_opencv_contrib=OFF",
        "-D", "BUILD_opencv_features2d=OFF",
        "-D", "BUILD_opencv_flann=OFF",
        "-D", "BUILD_opencv_gapi=OFF",
        "-D", "WITH_CUDA=OFF",
        "-D", "WITH_CUDNN=OFF",
        "-D", "OPENCV_DNN_CUDA=OFF",
        "-D", "ENABLE_FAST_MATH=1",
        "-D", "WITH_CUBLAS=0",
        "-D", "BUILD_opencv_gpu=OFF",
        "-D", "BUILD_opencv_ml=OFF",
        "-D", "BUILD_opencv_nonfree=OFF",
        "-D", "BUILD_opencv_objdetect=OFF",
        "-D", "BUILD_opencv_photo=OFF",
        "-D", "BUILD_opencv_stitching=OFF",
        "-D", "BUILD_opencv_superres=OFF",
        "-D", "BUILD_opencv_ts=OFF",
        "-D", "BUILD_opencv_video=OFF",
        "-D", "BUILD_videoio_plugins=OFF",
        "-D", "BUILD_opencv_videostab=OFF",
        "-D", "WITH_IPP=ON",
        "-D", "WITH_MKL=ON",        # 启用 Intel MKL
        "-D", "MKL_USE_TBB=ON",
        "-D", "WITH_TBB=ON",
        "-D", "BUILD_TBB=ON",
        "-D", "WITH_TURBOJPEG=ON",
        "-D", "WITH_LAPACK=ON",
        "-D", "WITH_BLAS=ON",
        ".."
    ], check=True)
    subprocess.run(["make", "-j4"], check=True)
    subprocess.run(["make", "install"], check=True)
    cache_header = os.path.join(cache_dir, "include/opencv4/")
    cache_lib = os.path.join(cache_dir, "lib/")
    return cache_header, cache_lib

def _build_lib(name):
    logger.warning(
        f'Pre-built library not found for {name}, starting JIT compilation')
    if name == "torchpipe_core":
        # python -m omniback.utils.build_lib --source-dirs csrc/torchplugins/ csrc/helper/ --include-dirs=csrc/ --build-with-cuda --name torchpipe_core
        subprocess.run(
            [
                sys.executable,
                "-m",
                "omniback.utils.build_lib",
                "--source-dirs",
                os.path.join(csrc_dir, "csrc/torchplugins/"),
                os.path.join(csrc_dir, "csrc/helper/"),
                "--include-dirs",
                os.path.join(csrc_dir, "csrc/"),
                "--build-with-cuda",
                "--name",
                "torchpipe_core"
            ],
            check=True,
            env={**os.environ, "EXAMPLE_ENV": "1"},
        )
    elif name == "torchpipe_nvjpeg":
        # python -m omniback.utils.build_lib --source-dirs csrc/nvjpeg_torch/ --include-dirs=csrc/ --build-with-cuda --ldflags="-lnvjpeg" --name torchpipe_nvjpeg
        subprocess.run(
            [
                sys.executable,
                "-m",
                "omniback.utils.build_lib",
                "--source-dirs",
                os.path.join(csrc_dir, "csrc/nvjpeg_torch/"),
                "--include-dirs",
                os.path.join(csrc_dir, "csrc/"),
                "--build-with-cuda",
                "--ldflags=-lnvjpeg",
                "--name",
                "torchpipe_nvjpeg"
            ],
            check=True,
            # TVM_FFI_DISABLE_TORCH_C_DLPACK
            env={**os.environ, "EXAMPLE_ENV": "1"},
        )
    elif name == "torchpipe_tensorrt":
        # python -m omniback.utils.build_lib --source-dirs csrc/tensorrt_torch/ --include-dirs=csrc/ --build-with-cuda --ldflags="-lnvinfer -lnvonnxparser  -lnvinfer_plugin" --name torchpipe_tensorrt
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
                "--ldflags=-lnvinfer -lnvonnxparser -lnvinfer_plugin",
                "--name",
                "torchpipe_tensorrt"
            ],
            check=True,
            # TVM_FFI_DISABLE_TORCH_C_DLPACK
            env={**os.environ, "EXAMPLE_ENV": "1"},
        )
 
    elif name == "torchpipe_opencv":
        # python -m omniback.utils.build_lib --no-torch --source-dirs csrc/mat_torch/ --include-dirs csrc/ /usr/local/include/opencv4/ --ldflags "-lopencv_core -lopencv_imgproc -lopencv_imgcodecs" --name torchpipe_opencv
        cv_inc, cv_lib = get_cv_include_lib_dir()
        if cv_inc is None:
            cv_inc, cv_lib = cache_cv_dir()
        if cv_inc is None:
            raise RuntimeError(
                "can not find opencv. set it through OPENCV_INCLUDE && OPENCV_LIB")
            
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
        raise RuntimeError(f"Unsupported lib: {name}")


def _load_or_build_lib_skip_if_error(name):
    try:
        return _load_or_build_lib(name)
    except Exception as e:
        logger.warning(
            f'Failed to load or JIT compile `{name}` extensions: \n{e}')

        
def _load_or_build_lib(name):
    if not _load_lib(name):
        _build_lib(name)
        return _load_lib(name)
