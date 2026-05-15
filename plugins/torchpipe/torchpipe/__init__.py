# isort: skip_file
"""TorchPipe: PyTorch-native inference serving library."""

from __future__ import annotations

import ctypes
import logging
import os
from importlib.metadata import version as _get_version

import torch
from packaging import version

import omniback

logger = logging.getLogger(__name__)

# Store original environment variable for later restoration
_ORI_TVM_FFI_DISABLE_TORCH_C_DLPACK = os.environ.get("TVM_FFI_DISABLE_TORCH_C_DLPACK", "0")
if _ORI_TVM_FFI_DISABLE_TORCH_C_DLPACK == "0":
    os.environ["TVM_FFI_DISABLE_TORCH_C_DLPACK"] = "1"

# Version information
try:
    __version__ = _get_version("torchpipe")
except Exception as e:
    logger.debug("Failed to get torchpipe version: %s", e)
    __version__ = "0.0.0-dev"

# -----------------------
# assert omniback.compiled_with_cxx11_abi() == torch.compiled_with_cxx11_abi()

logger.debug("torch.cuda.is_available() = %s", torch.cuda.is_available())

# Ensure PyTorch uses optimal thread settings
try:
    torch.set_num_threads(torch.get_num_threads())
except Exception as e:
    logger.warning("Failed to set PyTorch threads: %s", e)

# -----------------------
# Lazy imports to avoid circular dependencies
try:
    from .load_libs import _load_or_build_lib, _load_or_build_lib_skip_if_error  # noqa: E402
    from .load_libs import _setting_group_handle  # noqa: E402
    _HAS_LOAD_LIBS = True
except ImportError as e:
    logger.error("Failed to import load_libs: %s", e)
    _HAS_LOAD_LIBS = False
    _load_or_build_lib = None
    _load_or_build_lib_skip_if_error = None
    _setting_group_handle = None

# Environment-based feature flags
SKIP_ALL = os.environ.get("TORCHPIPE_SKIP_ALL", "0")
_extensions_loaded = False

def _load_extensions() -> None:
    """Load or JIT-compile TorchPipe extensions.

    This function is idempotent and can be called multiple times safely.
    """
    global SKIP_ALL
    global _extensions_loaded

    if _extensions_loaded:
        logger.debug("Extensions already loaded, skipping")
        return

    if not _HAS_LOAD_LIBS:
        logger.warning("load_libs module not available, skipping extension loading")
        SKIP_ALL = "1"
        return

    if SKIP_ALL == "1":
        logger.debug("TORCHPIPE_SKIP_ALL is set, skipping extension loading")
        return

    try:
        logger.info("Loading TorchPipe core extensions...")
        _load_or_build_lib("torchpipe_core")
        if torch.cuda.is_available():
            _load_or_build_lib("torchpipe_core_cuda")
        logger.debug("Core extensions loaded successfully")
    except Exception as e:
        logger.warning("Failed to load or JIT compile builtin extensions: %s", e, exc_info=True)
        SKIP_ALL = "1"
        return

    # Load CUDA-dependent extensions
    if torch.cuda.is_available():
        skip_tensorrt = os.environ.get("TORCHPIPE_SKIP_TENSORRT", "0")
        if skip_tensorrt != "1":
            try:
                _load_or_build_lib_skip_if_error("torchpipe_tensorrt")
                logger.debug("TensorRT extension loaded")
            except Exception as e:
                logger.debug("TensorRT extension not loaded: %s", e)
        try:
            _load_or_build_lib_skip_if_error("torchpipe_nvjpeg")
            logger.debug("NVJPEG extension loaded")
        except Exception as e:
            logger.debug("NVJPEG extension not loaded: %s", e)
    else:
        logger.debug("[JIT] CUDA is not available, skip loading CUDA extensions.")

    # Load OpenCV extension
    skip_opencv = os.environ.get("TORCHPIPE_SKIP_OPENCV", "0")
    if skip_opencv != "1":
        try:
            _load_or_build_lib_skip_if_error("torchpipe_opencv")
            logger.debug("OpenCV extension loaded")
        except Exception as e:
            logger.debug("OpenCV extension not loaded: %s", e)

    _extensions_loaded = True
    logger.info("TorchPipe extensions loaded successfully")


if _HAS_LOAD_LIBS:
    _load_extensions()

# Load backend group configuration
_grp_config = os.path.join(os.path.dirname(__file__), "group-torchpipe.toml")
if not os.path.exists(_grp_config):
    logger.error("Group config not found: %s", _grp_config)
    raise RuntimeError(f"Group config not found: {_grp_config}")

if _HAS_LOAD_LIBS and _setting_group_handle:
    try:
        _setting_group_handle(_grp_config)
        logger.debug("Loaded group config from %s", _grp_config)
    except Exception as e:
        logger.error("Failed to load group config: %s", e, exc_info=True)
        raise


# Re-export core functionality
pipe = omniback.pipe
Dict = omniback.Dict
register = omniback.register


def set_fast_dlpack() -> None:
    """Enable fast DLPack exchange using PyTorch C API.

    This provides zero-copy tensor exchange between PyTorch and TVM FFI.
    """
    try:
        import tvm_ffi

        tvm_ffi._optional_torch_c_dlpack.load_torch_c_dlpack_extension()
        tvm_ffi._optional_torch_c_dlpack.patch_torch_cuda_stream_protocol()

        if not hasattr(torch.Tensor, "__dlpack_c_exchange_api__"):
            logger.debug("torch.Tensor.__dlpack_c_exchange_api__ not available")
            return

        api_attr = torch.Tensor.__dlpack_c_exchange_api__
        if not api_attr:
            logger.debug("torch.Tensor.__dlpack_c_exchange_api__ is None")
            return

        # PyCapsule - extract the pointer as integer
        pythonapi = ctypes.pythonapi
        # Set restype to c_size_t to get integer directly (avoids c_void_p quirks)
        pythonapi.PyCapsule_GetPointer.restype = ctypes.c_size_t
        pythonapi.PyCapsule_GetPointer.argtypes = [ctypes.py_object, ctypes.c_char_p]

        capsule_name = b"dlpack_exchange_api"
        api_ptr = pythonapi.PyCapsule_GetPointer(api_attr, capsule_name)

        if api_ptr == 0:
            raise RuntimeError("API pointer from PyCapsule should not be NULL")

        omniback.ffi.set_dlpack_exchange_api(api_ptr)
        logger.debug("Fast DLPack enabled successfully")
    except Exception as e:
        logger.warning("Failed to enable fast DLPack: %s", e)
        raise


# Enable fast DLPack for PyTorch >= 2.3.0
try:
    if version.parse(torch.__version__) >= version.parse("2.3.0"):
        if _ORI_TVM_FFI_DISABLE_TORCH_C_DLPACK == "0":
            os.environ["TVM_FFI_DISABLE_TORCH_C_DLPACK"] = "0"
        set_fast_dlpack()
except Exception as e:
    logger.warning("Failed to enable fast DLPack for PyTorch 2.3+: %s", e)


# Clean up private variables from namespace
try:
    del _ORI_TVM_FFI_DISABLE_TORCH_C_DLPACK, _grp_config, _HAS_LOAD_LIBS
except NameError:
    pass  # Some variables may not be defined if loading failed

from . import backends

__all__ = ["pipe", "Dict", "register", "set_fast_dlpack", "__version__", "backends"]
