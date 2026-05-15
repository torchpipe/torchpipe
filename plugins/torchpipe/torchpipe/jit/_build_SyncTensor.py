"""Build SyncTensor Addon."""

from __future__ import annotations


import argparse
import os
import shutil
import subprocess
import sys
import sysconfig
import tempfile
from collections.abc import Sequence
from pathlib import Path

import torch
import torch.torch_version
import torch.utils.cpp_extension

assert torch.cuda.is_available(), "CUDA is required to build the extension."

import omniback
import tvm_ffi
# grp = "torchpipe.core_cuda"
grp = "om"

omniback.ffi.partial_register("SyncTensor", grp, 0)

@tvm_ffi.register_global_func(f"{grp}.SyncTensor.init")
def SyncTensor_init(self: om.Dict, params: dict[str, str], options: om.Dict):
    print("SyncTensor_init", self,  params, options)
    # get_current_dependency
    if torch.cuda.current_stream() != torch.cuda.default_stream():
        return

    context = tvm_ffi.use_torch_stream(torch.cuda.stream(torch.cuda.Stream()))
    context.__enter__()
    self["context"] = context

# @tvm_ffi.register_global_func(f"{grp}.SyncTensor.forward")
# def SyncTensor_forward(self: om.Dict, ios: list(om.Dict)):
#     print("SyncTensor_forward",self,  params, options)
#     if torch.cuda.current_stream() != torch.cuda.default_stream():
#         return

#     context = tvm_ffi.use_torch_stream(torch.cuda.stream(torch.cuda.Stream()))
#     context.__enter__()
#     self["context"] = context

print("SyncTensor register")
