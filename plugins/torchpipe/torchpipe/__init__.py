# from ._C import *
import torch

torch.cuda.init()

import hami

if (hami._C.use_cxx11_abi() != torch._C._GLIBCXX_USE_CXX11_ABI):
    info  = f"Incompatible C++ ABI detected. Please re-install PyTorch/Torchpipe or hami with the same C++ ABI. "
    info += "hami CXX11_ABI = {}, torch CXX11_ABI = {}. ".format(hami._C.use_cxx11_abi(), torch._C._GLIBCXX_USE_CXX11_ABI)
    info += """\nFor hami, you can use 
        pip3 install hami-core --platform manylinux2014_x86_64 --only-binary=:all:   --target `python3 -c "import site; print(site.getsitepackages()[0])"` 
        to install the pre-cxx11 abi version.
    """
    raise RuntimeError(info)
from .extension import _load_library

# from . import _C

# from . import native
_load_library("native")
_load_library("image")
_load_library("mat")
_load_library("trt")

# try:
#     _load_library("_C")
#     _HAS_GPU_VIDEO_DECODER = True
# except (ImportError, OSError):
#     _HAS_GPU_VIDEO_DECODER = False