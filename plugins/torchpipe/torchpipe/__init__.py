# from ._C import *

# from .version import __version__

import omniback

import torch

from importlib.metadata import version

__version__ = version("torchpipe")

from . import libnative, libimage, libmat, libtrt

from . import utils
if (omniback._C.use_cxx11_abi() != torch._C._GLIBCXX_USE_CXX11_ABI):
    info = f"Incompatible C++ ABI detected. Please re-install PyTorch/Torchpipe or omniback with the same C++ ABI. "
    info += "omniback CXX11_ABI = {}, torch CXX11_ABI = {}. ".format(
        omniback._C.use_cxx11_abi(), torch._C._GLIBCXX_USE_CXX11_ABI)
    info += f"""\nFor omniback, you can use 
        pip3 install omniback --platform manylinux2014_x86_64 --only-binary=:all:   --target `python3 -c "import site; print(site.getsitepackages()[0])"` 
        to install the pre-cxx11 abi version. Or use `USE_CXX11_ABI={int(not omniback._C.use_cxx11_abi())} pip install -e .` to rebuild omniback.
    """
    raise RuntimeError(info)
from .extension import _load_library


torch.cuda.init()



# _load_library("native")
# _load_library("image")
# _load_library("mat")
# _load_library("trt")

# try:
#     _load_library("_C")
#     _HAS_GPU_VIDEO_DECODER = True
# except (ImportError, OSError):
#     _HAS_GPU_VIDEO_DECODER = False

