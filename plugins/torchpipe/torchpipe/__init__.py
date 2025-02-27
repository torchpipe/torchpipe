# from ._C import *
import torch

torch.cuda.init()

import hami
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