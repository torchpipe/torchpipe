# isort: skip_file

import logging
logger = logging.getLogger(__name__)  # type: ignore

import ctypes, os

ORI_TVM_FFI_DISABLE_TORCH_C_DLPACK = os.environ.get(
    "TVM_FFI_DISABLE_TORCH_C_DLPACK", "0")
if ORI_TVM_FFI_DISABLE_TORCH_C_DLPACK == "0":
    os.environ["TVM_FFI_DISABLE_TORCH_C_DLPACK"] = "1"

# import time
# start = time.time()
import omniback
# print(f'end = {time.time() - start}')

import torch

# from importlib.metadata import version

# __version__ = version("torchpipe")


try:
    # type: ignore[import-not-found]
    from ._version import __version__, __version_tuple__
except ImportError:
    __version__ = "0.0.0.dev0"
    __version_tuple__ = (0, 0, 0, "dev0", "unknown")

 
# -----------------------

from . import utils ## sort off
assert omniback.compiled_with_cxx11_abi() == torch.compiled_with_cxx11_abi()

logger.info(f'torch.cuda.is_available() = {torch.cuda.is_available()}')

torch.set_num_threads(torch.get_num_threads())

# -----------------------
from .load_libs import _load_or_build_lib 

try:
    _load_or_build_lib("torchpipe_core")
    _load_or_build_lib("torchpipe_nvjpeg")
    _load_or_build_lib("torchpipe_opencv")
except Exception as e:
    logger.warning(f'Failed to load or JIT compile builtin extensions: \n{e}')


# -----------------------
pipe = omniback.pipe
Dict = omniback.Dict
register = omniback.register


# -----------------------
def set_fast_dlpack():
    import tvm_ffi
    tvm_ffi._optional_torch_c_dlpack.load_torch_c_dlpack_extension()
    tvm_ffi._optional_torch_c_dlpack.patch_torch_cuda_stream_protocol()
    if hasattr(torch.Tensor, "__dlpack_c_exchange_api__"):
        # type: ignore[attr-defined]
        api_attr = torch.Tensor.__dlpack_c_exchange_api__
        if api_attr:
            # PyCapsule - extract the pointer as integer
            pythonapi = ctypes.pythonapi
            # Set restype to c_size_t to get integer directly (avoids c_void_p quirks)
            pythonapi.PyCapsule_GetPointer.restype = ctypes.c_size_t
            pythonapi.PyCapsule_GetPointer.argtypes = [
                ctypes.py_object, ctypes.c_char_p]
            capsule_name = b"dlpack_exchange_api"
            api_ptr = pythonapi.PyCapsule_GetPointer(api_attr, capsule_name)
            assert api_ptr != 0, "API pointer from PyCapsule should not be NULL"
            omniback.ffi.set_dlpack_exchange_api(api_ptr)


if ORI_TVM_FFI_DISABLE_TORCH_C_DLPACK == "0":
    os.environ["TVM_FFI_DISABLE_TORCH_C_DLPACK"] = "0"
    
set_fast_dlpack()
