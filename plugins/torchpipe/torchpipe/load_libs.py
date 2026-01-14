from omniback.utils import build_lib
import ctypes, os, sys
import logging
logger = logging.getLogger(__name__)  # type: ignore
import subprocess

csrc_dir = os.path.dirname(__file__)
def load_whl_lib(path_of_cache):
    p = os.path.join(os.path.dirname(__file__),
                     os.path.basename(path_of_cache))
    if os.path.exists(p):
        ctypes.CDLL(p, mode=ctypes.RTLD_GLOBAL)
        return True
    return False

def _load_lib(name):
    if name == "torchpipe_core":
        torchpipe_core = build_lib.get_cache_lib(
            name, "cuda", False)
        if load_whl_lib(torchpipe_core):
            return True
        if os.path.exists(torchpipe_core):
            ctypes.CDLL(torchpipe_core, mode=ctypes.RTLD_GLOBAL)
            return True
        
    elif name == "torchpipe_nvjpeg":
        torchpipe_nvjpeg = build_lib.get_cache_lib(
            "torchpipe_nvjpeg", "cuda", False)
        if load_whl_lib(torchpipe_nvjpeg):
            return True
        if os.path.exists(torchpipe_nvjpeg):
            ctypes.CDLL(torchpipe_nvjpeg, mode=ctypes.RTLD_GLOBAL)
            return True
        
    elif name == "torchpipe_opencv":
        torchpipe_opencv = build_lib.get_cache_lib(
            "torchpipe_opencv", "", True)
        if load_whl_lib(torchpipe_opencv):
            return True
        if os.path.exists(torchpipe_opencv):
            ctypes.CDLL(torchpipe_opencv, mode=ctypes.RTLD_GLOBAL)
            return True
    
    return False

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
 
    elif name == "torchpipe_opencv":
        # python -m omniback.utils.build_lib --no-torch --source-dirs csrc/mat_torch/ --include-dirs csrc/ /usr/local/include/opencv4/ --ldflags "-lopencv_core -lopencv_imgproc -lopencv_imgcodecs" --name torchpipe_opencv
        subprocess.run(
            [
                sys.executable,
                "-m",
                "omniback.utils.build_lib",
                "--source-dirs",
                os.path.join(csrc_dir, "csrc/mat_torch/"),
                "--include-dirs",
                os.path.join(csrc_dir, "csrc/"),
                f"/usr/local/include/opencv4/",
                "--build-with-cuda",
                "--no-torch",
                "--ldflags=-lopencv_core -lopencv_imgproc -lopencv_imgcodecs",
                "--name",
                "torchpipe_opencv"
            ],
            check=True,
            env={**os.environ, "EXAMPLE_ENV": "1"},
        )
    else:
        raise RuntimeError(f"Unsupported lib: {name}")

def _load_or_build_lib(name):
    if not _load_lib(name):
        _build_lib(name)
        return _load_lib(name)
