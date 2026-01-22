import subprocess
from pathlib import Path
from omniback.utils import build_lib
from .utils._cache_setting import get_cache_dir

import ctypes, os, sys
import logging
logger = logging.getLogger(__name__)  # type: ignore

csrc_dir = os.path.dirname(__file__)

def load_whl_lib(path_of_cache):
    p = os.path.join(os.path.dirname(__file__), 'lib',
                     os.path.basename(path_of_cache))
    if os.path.exists(p):
        ctypes.CDLL(p, mode=ctypes.RTLD_GLOBAL)
        logger.info(f'Successfully loaded precompiled {p} from the installed package')
        return True
    return False

def get_whl_lib(path_of_cache):
    p = os.path.join(os.path.dirname(__file__), 'lib',
                     os.path.basename(path_of_cache))
    if os.path.exists(p):
        # ctypes.CDLL(p, mode=ctypes.RTLD_GLOBAL)
        # logger.info(f'Successfully loaded precompiled {p} from the installed package')
        return p
    return None

def _load_lib_with_torch_cuda(name):
    device = f"cuda{torch.version.cuda.split('.')[0]}"
    local_lib = build_lib.get_cache_lib(
        name, device, False)
    if load_whl_lib(local_lib):
        return True
    if name == "torchpipe_tensorrt":
        if os.path.exists(local_lib):
            try:
                ctypes.CDLL(local_lib, mode=ctypes.RTLD_GLOBAL)
            except OSError:
                from .utils._build_trt import get_trt_include_lib_dir
                _, lib_dir = get_trt_include_lib_dir()
                if lib_dir is None:
                    import torch
                    cuda_version = int(torch.version.cuda.split('.')[0])

                    logger.warning(
                        f"Can not find TensorRT. Skip load torchpipe_tensorrt. Set TENSORRT_INCLUDE and TENSORRT_LIB")
                    #  or `pip install tensorrt-cu{cuda_version}`
                    has_import_trt=False
                    try:
                        import tensorrt
                        has_import_trt = True
                    except:
                        return False
                    if has_import_trt:
                        ctypes.CDLL(local_lib, mode=ctypes.RTLD_GLOBAL)
                        return True
                        
                os.environ["LD_LIBRARY_PATH"] = f"{lib_dir}:" + \
                    os.environ.get("LD_LIBRARY_PATH", "")
                nvinfer = Path(lib_dir)/"libnvinfer.so"
                nvonnxparser = Path(lib_dir)/"libnvonnxparser.so"
                nvinfer_plugin = Path(lib_dir)/"libnvinfer_plugin.so"

                ctypes.CDLL(nvinfer.resolve(), mode=ctypes.RTLD_GLOBAL)
                ctypes.CDLL(nvonnxparser.resolve(), mode=ctypes.RTLD_GLOBAL)
                ctypes.CDLL(nvinfer_plugin.resolve(), mode=ctypes.RTLD_GLOBAL)
                
                ctypes.CDLL(local_lib, mode=ctypes.RTLD_GLOBAL)
            return True
    else:
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
                from .utils._build_cv import get_cv_include_lib_dir
                _, lib_dir = get_cv_include_lib_dir()
                os.environ["LD_LIBRARY_PATH"] = f"{lib_dir}:" + \
                    os.environ.get("LD_LIBRARY_PATH", "")
                # if lib_dir is None:
                #     raise RuntimeError("can not find opencv library. You can set it through OPENCV_LIB")
                core = Path(lib_dir)/"libopencv_core.so"
                imgproc = Path(lib_dir)/"libopencv_imgproc.so"
                imgcodecs = Path(lib_dir)/"libopencv_imgcodecs.so"
                
                ctypes.CDLL(core.resolve(), mode=ctypes.RTLD_GLOBAL)
                ctypes.CDLL(imgproc.resolve(), mode=ctypes.RTLD_GLOBAL)
                ctypes.CDLL(imgcodecs.resolve(), mode=ctypes.RTLD_GLOBAL)
                    
                ctypes.CDLL(torchpipe_opencv, mode=ctypes.RTLD_GLOBAL)
            return True
    else:
        return _load_lib_with_torch_cuda(name)
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
    elif name == "torchpipe_tensorrt":
        # python -m omniback.utils.build_lib --source-dirs csrc/tensorrt_torch/ --include-dirs=csrc/ --build-with-cuda --ldflags="-lnvinfer -lnvonnxparser  -lnvinfer_plugin" --name torchpipe_tensorrt
        from .utils._build_trt import _build_trt
        _build_trt(csrc_dir)
 
    elif name == "torchpipe_opencv":
        # python -m omniback.utils.build_lib --no-torch --source-dirs csrc/mat_torch/ --include-dirs csrc/ /usr/local/include/opencv4/ --ldflags "-lopencv_core -lopencv_imgproc -lopencv_imgcodecs" --name torchpipe_opencv
        from .utils._build_cv import _build_cv
        _build_cv(csrc_dir)
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

if __name__ == "__main__":
    import fire
    fire.Fire({
        "build":  _build_lib
    })