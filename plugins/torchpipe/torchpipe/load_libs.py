"""Library loading and JIT compilation utilities for TorchPipe."""

from __future__ import annotations

import ctypes
import glob
import importlib.util
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

import tvm_ffi
from omniback.utils import build_lib

from .utils._cache_setting import get_cache_dir

logger = logging.getLogger(__name__)

csrc_dir = os.path.dirname(__file__)
current_dir = os.path.dirname(__file__)


def load_whl_lib(path_of_cache: str, symbol_global: bool = True) -> bool:
    """Load pre-compiled library from wheel package.

    Args:
        path_of_cache: Path to the cached library
        symbol_global: Whether to use RTLD_GLOBAL mode

    Returns:
        True if library was loaded successfully
    """
    p = os.path.join(os.path.dirname(__file__), 'lib',
                     os.path.basename(path_of_cache))
    if not os.path.exists(p):
        return False

    mode = ctypes.RTLD_GLOBAL if symbol_global else ctypes.RTLD_LOCAL
    ctypes.CDLL(p, mode=mode)
    logger.info('Successfully loaded precompiled %s from the installed package', p)
    return True


def get_whl_lib(path_of_cache: str) -> str | None:
    """Get path to pre-compiled library in wheel package.

    Args:
        path_of_cache: Path to the cached library

    Returns:
        Path to library if exists, None otherwise
    """
    p = os.path.join(os.path.dirname(__file__), 'lib',
                     os.path.basename(path_of_cache))
    return p if os.path.exists(p) else None


def get_current_rpath(so_path: str) -> str:
    """Get current RPATH of a shared library using patchelf.

    Args:
        so_path: Path to shared library

    Returns:
        Current RPATH string, empty if failed
    """
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


def fix_nvinfer_rpath(path: str) -> None:
    """Fix RPATH for TensorRT libraries in the given directory.

    Args:
        path: Directory containing TensorRT libraries
    """
    patchelf_path = shutil.which("patchelf")
    if patchelf_path is None:
        logger.warning("patchelf not found; skipping RPATH fix")
        logger.warning(
            '[JIT] You may need to:\n'
            'export LD_LIBRARY_PATH=%s:$LD_LIBRARY_PATH', path
        )
        return

    for library in glob.iglob(os.path.join(path, "*.so*")):
        basename = os.path.basename(library)
        if not basename.startswith("libnvinfer.so"):
            continue

        if get_current_rpath(library) != "":
            continue

        try:
            subprocess.run(
                [patchelf_path, "--set-rpath", "$ORIGIN", library],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                check=True
            )
            logger.debug("Successfully set RPATH for %s", library)
        except subprocess.CalledProcessError as e:
            logger.warning("Failed to run patchelf on %s: %s (stderr: %s)",
                           library, e, e.stderr)
        except Exception as e:
            logger.warning("Unexpected error running patchelf on %s: %s", library, e)


def try_load_libs_from_dir(path: str) -> None:
    """Attempt to load all libraries from a directory.

    Args:
        path: Directory containing libraries
    """
    for pattern in ["*.so*", "*.dll*"]:
        for lib in glob.iglob(os.path.join(path, pattern)):
            try:
                ctypes.CDLL(lib, mode=ctypes.RTLD_GLOBAL)
            except OSError as e:
                logger.debug("Failed to load library %s: %s", lib, e)


def _load_lib_with_torch(name: str, device: str = "cuda") -> bool:
    """Load library with torch-specific handling.

    Args:
        name: Library name
        device: Device type (cpu/cuda)

    Returns:
        True if library was loaded successfully
    """
    local_lib = build_lib.get_cache_lib(name, device, False)

    if load_whl_lib(local_lib):
        return True

    if not os.path.exists(local_lib):
        return False

    if name == "torchpipe_tensorrt":
        # Check if we should skip TensorRT entirely
        if os.environ.get("TORCHPIPE_SKIP_TENSORRT", "0") == "1":
            logger.debug("TORCHPIPE_SKIP_TENSORRT=1, skipping torchpipe_tensorrt load")
            return False
            
        from .utils._build_trt import get_trt_include_lib_dir
        _, lib_dir = get_trt_include_lib_dir()

        if lib_dir is not None:
            fix_nvinfer_rpath(lib_dir)

        try:
            ctypes.CDLL(local_lib, mode=ctypes.RTLD_GLOBAL)
        except OSError as e:
            logger.debug("Failed to load torchpipe_tensorrt directly: %s", e)
            if lib_dir is None:
                import torch
                logger.warning(
                    "Cannot find TensorRT. Skip load torchpipe_tensorrt. "
                    "Set TENSORRT_INCLUDE and TENSORRT_LIB, or set TORCHPIPE_SKIP_TENSORRT=1"
                )
                return False
            else:
                try_load_libs_from_dir(lib_dir)
                try:
                    ctypes.CDLL(local_lib, mode=ctypes.RTLD_GLOBAL)
                except OSError as e2:
                    logger.warning(
                        "Cannot load torchpipe_tensorrt after loading TensorRT libraries. "
                        "Error: %s. You may need to set LD_LIBRARY_PATH=%s:$LD_LIBRARY_PATH",
                        e2, lib_dir
                    )
                    raise

        return True

    ctypes.CDLL(local_lib, mode=ctypes.RTLD_GLOBAL)
    return True


def _load_lib(name: str) -> bool:
    """Load a specific library by name.

    Args:
        name: Library name (e.g., torchpipe_core, torchpipe_opencv)

    Returns:
        True if library was loaded successfully
    """
    if name == "torchpipe_opencv":
        return _load_opencv_lib()
    elif name == "torchpipe_core":
        return _load_lib_with_torch(name, device="cpu")
    else:
        return _load_lib_with_torch(name, device="cuda")


def _load_opencv_lib() -> bool:
    """Load OpenCV library with special handling for dependencies."""
    if os.environ.get("TORCHPIPE_SKIP_OPENCV", "0") == "1":
        logger.info("TORCHPIPE_SKIP_OPENCV is set, skipping OpenCV library loading")
        return False
    
    torchpipe_opencv = build_lib.get_cache_lib("torchpipe_opencv", "", True)

    if load_whl_lib(torchpipe_opencv, symbol_global=False):
        return True

    if not os.path.exists(torchpipe_opencv):
        return False

    try:
        ctypes.CDLL(torchpipe_opencv, mode=ctypes.RTLD_LOCAL)
        return True
    except OSError as e:
        logger.debug(f"Failed to load torchpipe_opencv directly: {e}")
        logger.info("Attempting to load with OpenCV dependencies...")
        
        from .utils._build_cv import get_cv_include_lib_dir, is_system_exists_cv, get_system_cv

        cv_inc, lib_dir = get_cv_include_lib_dir()
        if lib_dir is None and is_system_exists_cv():
            _, lib_dir = get_system_cv()

        if lib_dir is None:
            logger.error(
                "Cannot find OpenCV library. You can set it through OPENCV_LIB "
                "environment variable, or install OpenCV system-wide, or set "
                "TORCHPIPE_SKIP_OPENCV=1 to skip OpenCV support."
            )
            return False

        os.environ["LD_LIBRARY_PATH"] = f"{lib_dir}:" + os.environ.get("LD_LIBRARY_PATH", "")

        # Load required OpenCV libraries with error handling
        required_libs = ["libopencv_core.so", "libopencv_imgproc.so", "libopencv_imgcodecs.so"]
        for lib_name in required_libs:
            lib_path = Path(lib_dir) / lib_name
            try:
                if lib_path.exists():
                    ctypes.CDLL(lib_path.resolve(), mode=ctypes.RTLD_GLOBAL)
                    logger.debug(f"Successfully loaded {lib_name}")
                else:
                    logger.warning(f"OpenCV library not found: {lib_path}")
            except OSError as lib_err:
                logger.warning(f"Failed to load {lib_name}: {lib_err}")

        try:
            ctypes.CDLL(torchpipe_opencv, mode=ctypes.RTLD_GLOBAL)
            logger.info("Successfully loaded torchpipe_opencv with OpenCV dependencies")
            return True
        except OSError as final_err:
            logger.error(f"Failed to load torchpipe_opencv even after loading dependencies: {final_err}")
            return False


def _build_lib(name: str) -> None:
    """JIT compile a library.

    Args:
        name: Library name to build

    Raises:
        RuntimeError: If library name is not supported
    """
    logger.warning(
        '[JIT] Pre-built library not found for %s, starting JIT compilation', name)

    builders = {
        "torchpipe_core": _build_core_lib,
        "torchpipe_core_cuda": _build_core_cuda_lib,
        "torchpipe_nvjpeg": _build_nvjpeg_lib,
        "torchpipe_tensorrt": _build_tensorrt_lib,
        "torchpipe_opencv": _build_opencv_lib,
    }

    builder = builders.get(name)
    if builder is None:
        raise RuntimeError(f"Unsupported lib: {name}")

    builder()


def _run_build_command(args: list[str]) -> None:
    """Execute build command with proper environment."""
    subprocess.run(
        [sys.executable, "-m", "omniback.utils.build_lib"] + args,
        check=True,
        env={**os.environ, "EXAMPLE_ENV": "1"},
    )


def _build_core_lib() -> None:
    """Build torchpipe_core library."""
    _run_build_command([
        "--source-dirs",
        os.path.join(csrc_dir, "csrc/torchplugins/"),
        os.path.join(csrc_dir, "csrc/helper/"),
        "--include-dirs",
        os.path.join(csrc_dir, "csrc/"),
        "--name",
        "torchpipe_core"
    ])


def _build_core_cuda_lib() -> None:
    """Build torchpipe_core_cuda library."""
    _run_build_command([
        "--source-dirs",
        os.path.join(csrc_dir, "csrc/core_cuda/"),
        os.path.join(csrc_dir, "csrc/helper_cuda/"),
        "--include-dirs",
        os.path.join(csrc_dir, "csrc/"),
        "--build-with-cuda",
        "--name",
        "torchpipe_core_cuda"
    ])


def _build_nvjpeg_lib() -> None:
    """Build torchpipe_nvjpeg library."""
    _run_build_command([
        "--source-dirs",
        os.path.join(csrc_dir, "csrc/nvjpeg_torch/"),
        "--include-dirs",
        os.path.join(csrc_dir, "csrc/"),
        "--build-with-cuda",
        "--ldflags=-lnvjpeg",
        "--name",
        "torchpipe_nvjpeg"
    ])


def _build_tensorrt_lib() -> None:
    """Build torchpipe_tensorrt library."""
    from .utils._build_trt import _build_trt
    _build_trt(csrc_dir)


def _build_opencv_lib() -> None:
    """Build torchpipe_opencv library."""
    from .utils._build_cv import _build_cv
    _build_cv(csrc_dir)


def _load_or_build_lib_skip_if_error(name: str) -> bool:
    """Try to load or build a library, log warning on failure.

    Args:
        name: Library name

    Returns:
        True if successful, False otherwise
    """
    try:
        return _load_or_build_lib(name)
    except Exception as e:
        logger.warning('Failed to load or JIT compile `%s` extensions: %s', name, e)
        return False


def _load_or_build_lib(name: str) -> bool:
    """Load library or build it if not available.

    Args:
        name: Library name

    Returns:
        True if library is available after operation
    """
    if not _load_lib(name):
        _build_lib(name)
        return _load_lib(name)
    return True


def _set_group_callbacks(backend: str, grp_name: str) -> list:
    """Set up callbacks for backend group registration.

    Args:
        backend: Backend name
        grp_name: Group name

    Returns:
        List of callback functions
    """
    callbacks = []
    callbacks.append(lambda: _load_or_build_lib_skip_if_error(
        grp_name.replace("torchpipe.", "torchpipe_")))

    module_path = os.path.join(current_dir, f"jit/_build_{backend}.py")
    if os.path.exists(module_path):
        spec = importlib.util.spec_from_file_location(f"_build_{backend}", module_path)
        module = importlib.util.module_from_spec(spec)
        callbacks.append(lambda: spec.loader.exec_module(module) or True)

    return callbacks


def _setting_group_handle(toml_path: str) -> None:
    """Configure backend groups from TOML file.

    Args:
        toml_path: Path to group configuration TOML file
    """
    from omniback.group_registry import toml2groups

    _backend_to_groups, _ = toml2groups(toml_path)
    _register_backend_group = tvm_ffi.get_global_func("omniback.register_backend_group")

    for backend, grp_names in _backend_to_groups.items():
        if len(grp_names) != 1:
            raise ValueError(f"backend {backend} has multiple groups: {grp_names}")

        grp_name = next(iter(grp_names))
        for callback in _set_group_callbacks(backend, grp_name):
            _register_backend_group(backend, grp_name, callback)


if __name__ == "__main__":
    import fire
    fire.Fire({"build": _build_lib})
