import subprocess
from pathlib import Path
from omniback.utils import build_lib
from .utils._cache_setting import get_cache_dir

import ctypes, os, sys
import logging
import tvm_ffi
import os, glob, shutil
import subprocess

import importlib.util

logger = logging.getLogger(__name__)  # type: ignore

csrc_dir = os.path.dirname(__file__)
current_dir = os.path.join(os.path.dirname(__file__))

def load_whl_lib(path_of_cache, symbol_global=True):
    p = os.path.join(os.path.dirname(__file__), 'lib',
                     os.path.basename(path_of_cache))
    # print(f"load_whl_lib: {p}")
    if os.path.exists(p):
        mode = ctypes.RTLD_GLOBAL if symbol_global else ctypes.RTLD_LOCAL
        ctypes.CDLL(p, mode=mode)
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

# def get_unvalid_sm():
#     props = torch.cuda.get_device_properties(torch.cuda.current_device())
#     sm_version = int(float(f"{props.major}.{props.minor}") *10)
#     unvalid_sm = set(['sm60', 'sm70', 'sm80', 'sm86', 'sm90', 'sm90', 'sm100', "sm120"])
#     unvalid_sm.remove(f'sm{sm_version}')
#     return unvalid_sm

def get_current_rpath(so_path: str) -> str:
    try:
        result = subprocess.run(
            ["patchelf", "--print-rpath", so_path],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""

def try_load(library: str) -> bool:
    """
    Attempt to load a shared library.
    If it's libnvinfer, first set its RPATH to its own directory using patchelf (if available).
    Returns True if successfully loaded, False otherwise.
    """
    if not os.path.exists(library):
        return False

    return True
    # 更健壮地判断是否为 TensorRT 主库
    basename = os.path.basename(library)
    if not basename.startswith("libnvinfer.so"):
        # 非 TensorRT 库，直接尝试加载
        try:
            ctypes.CDLL(library, mode=ctypes.RTLD_GLOBAL)
            return True
        except OSError as e:
            logger.debug(f"Failed to load library {library}: {e}")
            return False

    # 是 libnvinfer，尝试用 patchelf 修复 RPATH
    lib_dir = os.path.dirname(os.path.abspath(library))
    
    # 检查 patchelf 是否可用
    patchelf_path = shutil.which("patchelf")
    if patchelf_path is None:
        logger.warning("patchelf not found; skipping RPATH fix for %s", library)
        logger.warning(
            f'[JIT] You may need to:\n'
            f'export LD_LIBRARY_PATH={os.path.dirname(library)}:$LD_LIBRARY_PATH'
        )
    else:
        if get_current_rpath(library) != "":
            return
        try:
            # 使用 check=True 确保命令失败时抛出异常
            subprocess.run(
                [patchelf_path, "--set-rpath", '$ORIGIN', library],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True
            )
            logger.debug("Successfully set RPATH for %s to %s", library, lib_dir)
        except subprocess.CalledProcessError as e:
            logger.warning("Failed to run patchelf on %s: %s (stderr: %s)", library, e, e.stderr)
        except Exception as e:
            logger.warning("Unexpected error running patchelf on %s: %s", library, e)
    
    ctypes.CDLL(library, mode=ctypes.RTLD_GLOBAL)
    return True

def fix_nvinfer_rpath(path):
    for library in glob.iglob(os.path.join(path, "*.so*")):
        basename = os.path.basename(library)
        if not basename.startswith("libnvinfer.so"):
            continue
        
        patchelf_path = shutil.which("patchelf")
        if patchelf_path is None:
            logger.warning("patchelf not found; skipping RPATH fix for %s", library)
            logger.warning(
                f'[JIT] You may need to:\n'
                f'export LD_LIBRARY_PATH={os.path.dirname(library)}:$LD_LIBRARY_PATH'
            )
        else:
            if get_current_rpath(library) != "":
                return
            try:
                # 使用 check=True 确保命令失败时抛出异常
                subprocess.run(
                    [patchelf_path, "--set-rpath", "$ORIGIN", library],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    text=True
                )
                logger.debug("Successfully set RPATH for %s", library)
            except subprocess.CalledProcessError as e:
                logger.warning("Failed to run patchelf on %s: %s (stderr: %s)", library, e, e.stderr)
            except Exception as e:
                logger.warning("Unexpected error running patchelf on %s: %s", library, e)
        

        
def try_load_libs_from_dir(path):
    for lib in glob.iglob(os.path.join(path, "*.so*")):
        try_load(lib)
    for lib in glob.iglob(os.path.join(path, "*.dll*")):
        try_load(lib)

def _load_lib_with_torch(name, device = "cuda"):
    # import torch
    # device = f"cuda{torch.version.cuda.split('.')[0]}"
    
    local_lib = build_lib.get_cache_lib(
        name, device, False)
    if load_whl_lib(local_lib):
        return True
    if name == "torchpipe_tensorrt":
        from .utils._build_trt import get_trt_include_lib_dir
        _, lib_dir = get_trt_include_lib_dir()
        
        if os.path.exists(local_lib):
            if lib_dir is not None:
                fix_nvinfer_rpath(lib_dir)
            try:
                ctypes.CDLL(local_lib, mode=ctypes.RTLD_GLOBAL)
            except OSError:

                # lib_dir = "/home/nan/.venv/lib/python3.12/site-packages/tensorrt_libs/"
                # lib_dir = "/home/nan/tensorrt_install/lib"
                if lib_dir is None:
                    import torch
                    cuda_version = int(torch.version.cuda.split('.')[0])

                    logger.warning(
                        f"Can not find TensorRT. Skip load torchpipe_tensorrt. Set TENSORRT_INCLUDE and TENSORRT_LIB")
                    #  or `pip install tensorrt-cu{cuda_version}`
                    # has_import_trt=False
                    # try:
                    #     import tensorrt
                    #     has_import_trt = True
                    # except:
                    #     return False
                    # if has_import_trt:
                    #     ctypes.CDLL(local_lib, mode=ctypes.RTLD_GLOBAL)
                    #     return True
                else:
                    try_load_libs_from_dir(lib_dir)
                # libs = ["libnvinfer.so.10", "libnvonnxparser.so.10", "libnvinfer_plugin.so.10"]        

                # for lib in libs:
                #     full_lib = Path(lib_dir)/lib
                #     print(full_lib)
                #     if full_lib.exists():
                #         print(f"full_lib={full_lib}")
                #         ctypes.CDLL(full_lib.resolve(), mode=ctypes.RTLD_GLOBAL)
                
                ctypes.CDLL(local_lib, mode=ctypes.RTLD_GLOBAL)
            
            return True
    else:
        if os.path.exists(local_lib):
            ctypes.CDLL(local_lib, mode=ctypes.RTLD_GLOBAL)
            return True
    
    if name == "torchpipe_tensorrt" and lib_dir is not None:
        logger.warning(
            f'[JIT] You may need to:\n'
            f'export LD_LIBRARY_PATH={lib_dir}:$LD_LIBRARY_PATH'
        )
    return False

def _load_lib(name):
    if name == "torchpipe_opencv":
        torchpipe_opencv = build_lib.get_cache_lib(
            "torchpipe_opencv", "", True)

        if load_whl_lib(torchpipe_opencv, symbol_global=False):
            return True
        if os.path.exists(torchpipe_opencv):
            try:
                ctypes.CDLL(torchpipe_opencv, mode=ctypes.RTLD_LOCAL)
            except OSError:
                from .utils._build_cv import get_cv_include_lib_dir, is_system_exists_cv, get_system_cv
                _, lib_dir = get_cv_include_lib_dir()
                if lib_dir is None:
                    if is_system_exists_cv():
                        _, lib_dir = get_system_cv()
                        
                if lib_dir is None:
                    raise RuntimeError(
                        "can not find opencv library. You can set it through OPENCV_LIB")
                    
                os.environ["LD_LIBRARY_PATH"] = f"{lib_dir}:" + \
                    os.environ.get("LD_LIBRARY_PATH", "")
                
                core = Path(lib_dir)/"libopencv_core.so"
                imgproc = Path(lib_dir)/"libopencv_imgproc.so"
                imgcodecs = Path(lib_dir)/"libopencv_imgcodecs.so"
                
                ctypes.CDLL(core.resolve(), mode=ctypes.RTLD_GLOBAL)
                ctypes.CDLL(imgproc.resolve(), mode=ctypes.RTLD_GLOBAL)
                ctypes.CDLL(imgcodecs.resolve(), mode=ctypes.RTLD_GLOBAL)
                    
                ctypes.CDLL(torchpipe_opencv, mode=ctypes.RTLD_GLOBAL)
            return True
    elif name == "torchpipe_core":
        return _load_lib_with_torch(name, device="cpu")
    else:
        return _load_lib_with_torch(name, device="cuda")
    return False




    
def _build_lib(name):
    logger.warning(
        f'[JIT] Pre-built library not found for {name}, starting JIT compilation')
    if name == "torchpipe_core":
        # python -m omniback.utils.build_lib --source-dirs csrc/torchplugins/ csrc/helper/ --include-dirs=csrc/ --name torchpipe_core
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
                "--name",
                name
            ],
            check=True,
            env={**os.environ, "EXAMPLE_ENV": "1"},
        )
    elif name == "torchpipe_core_cuda":
        subprocess.run(
            [
                sys.executable,
                "-m",
                "omniback.utils.build_lib",
                "--source-dirs",
                os.path.join(csrc_dir, "csrc/core_cuda/"),
                os.path.join(csrc_dir, "csrc/helper_cuda/"),
                "--include-dirs",
                os.path.join(csrc_dir, "csrc/"),
                "--build-with-cuda",
                "--name",
                name
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
                name
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
        return False
        
def _load_or_build_lib(name):
    if not _load_lib(name):
        _build_lib(name)
        return _load_lib(name)
    return True

def _set_group_callbacks(backend, grp_name):
    callbacks = []
    callbacks.append(lambda: _load_or_build_lib_skip_if_error(grp_name.replace("torchpipe.", "torchpipe_")))
    
    module_path = os.path.join(current_dir, f"jit/_build_{backend}.py")
    # print(f'module_path={module_path}')
    if os.path.exists(module_path):
        spec = importlib.util.spec_from_file_location(f"_build_{backend}", module_path)
        module = importlib.util.module_from_spec(spec)
        callbacks.append(lambda: spec.loader.exec_module(module) or True)        
    
    return callbacks

def _setting_group_handle(toml_path: str):
    from omniback.group_registry import toml2groups
    _backend_to_groups, _ = toml2groups(toml_path)
    # todo:  dependency loading
    _register_backend_group = tvm_ffi.get_global_func(
        "omniback.register_backend_group")
    for backend, grp_names in _backend_to_groups.items():
        assert len(
            grp_names) == 1, f"backend {backend} has multiple groups: {grp_names}"
        grp_name = next(iter(grp_names))
        for callback in _set_group_callbacks(backend, grp_name):
            _register_backend_group(backend, grp_name, callback)
      
if __name__ == "__main__":
    import fire
    fire.Fire({
        "build":  _build_lib
    })