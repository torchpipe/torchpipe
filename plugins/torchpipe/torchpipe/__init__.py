# from ._C import *


import hami

import torch

if (hami._C.use_cxx11_abi() != torch._C._GLIBCXX_USE_CXX11_ABI):
    info  = f"Incompatible C++ ABI detected. Please re-install PyTorch/Torchpipe or hami with the same C++ ABI. "
    info += "hami CXX11_ABI = {}, torch CXX11_ABI = {}. ".format(hami._C.use_cxx11_abi(), torch._C._GLIBCXX_USE_CXX11_ABI)
    info += f"""\nFor hami, you can use 
        pip3 install hami-core --platform manylinux2014_x86_64 --only-binary=:all:   --target `python3 -c "import site; print(site.getsitepackages()[0])"` 
        to install the pre-cxx11 abi version. Or use `USE_CXX11_ABI={int(not hami._C.use_cxx11_abi())} pip install -e .` to rebuild hami.
    """
    raise RuntimeError(info)
from .extension import _load_library


torch.cuda.init()

_load_library("native")
_load_library("image")
_load_library("mat")
_load_library("trt")

# try:
#     _load_library("_C")
#     _HAS_GPU_VIDEO_DECODER = True
# except (ImportError, OSError):
#     _HAS_GPU_VIDEO_DECODER = False