# Copyright 2021-2024 NetEase.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import torch

import torch._C
# 检查库的版本号
assert(torch._C)
if torch.cuda.is_available():
    assert(torch.zeros((1)).cuda().is_cuda)

import logging

import os

if os.path.exists("./torchpipe"):
    logging.info(f"found 'torchpipe/' in current dir.")

from .version import __version__, __torch_version__

if __torch_version__ != torch.__version__:
    logging.warning(
        "PyTorch version mismatch when compiling TorchPipe. Expected version %s, but got version %s.",
        __torch_version__, torch.__version__
    )

try:
    from torchpipe.libipipe import Interpreter
except ImportError as e:
    str_error = str(e)
    if not "libcvcuda.so" in str_error:
        raise e
    print(f"import error(`{str_error}`), try to get libcvcuda: ")
    import subprocess
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    subprocess.check_output(["python", os.path.join(cur_dir,"./tool/get_cvcuda.py"), "--sm61"])
    dynamic_path = os.path.join(os.path.expanduser("~"),"./.cache/nvcv/opt/nvidia/cvcuda0/lib/x86_64-linux-gnu/")

    import ctypes
    nvcv_types = ctypes.cdll.LoadLibrary(os.path.join(dynamic_path, "libnvcv_types.so"))
    cvcuda = ctypes.cdll.LoadLibrary(os.path.join(dynamic_path, "libcvcuda.so"))




from torchpipe.libipipe import (
    Interpreter,
    parse_toml,
    TASK_DATA_KEY,
    TASK_RESULT_KEY,
    TASK_BOX_KEY,
    TASK_INFO_KEY,
    TASK_EVENT_KEY,
    TASK_NODE_NAME_KEY,
    # any,
    register_backend,
    register_filter,
    WITH_CUDA,
    WITH_OPENVINO,
    Status,
    list_backends
)
if WITH_CUDA:
    from torchpipe.libipipe import  infer_shape, supported_opset
from torchpipe.libipipe import Event,ThreadSafeKVStorage
from torchpipe.libipipe import encrypt

# from torchpipe.utils import test, cpp_extension, patform, models
import torchpipe.utils
from .python_api import pipe
Pipe = pipe

import logging


logging.info(f"torchpipe version {__version__}")
torch.set_num_threads(1)
logging.info("torch: set_num_threads = 1;")

cur_path = os.path.abspath(__file__)
opencv_path = os.path.join(cur_path, "./opencv_install")
ipipe_path = os.path.join(cur_path, "./")

dir_name = os.path.dirname(os.path.realpath(__file__))
if dir_name == os.path.join(os.path.realpath(os.getcwd()), "torchpipe"):
    message = (
        f"Torchpipe was imported within its own root folder ({dir_name}). "
        "This is not expected to work and may give errors."
    )
    import warnings

    warnings.warn(message)


# from torchpipe.tool import vis as vis
from  torchpipe.utils import Visual 

def show(config):
    return Visual(config).launch()
__all__ = [
    "Interpreter",
    "parse_toml",
    "TASK_DATA_KEY",
    "TASK_RESULT_KEY",
    "TASK_INFO_KEY",
    "pipe",
    'Pipe',
    "TASK_BOX_KEY",
    "TASK_NODE_NAME_KEY",
    "TASK_EVENT_KEY",
    "encrypt",
    # "any",
    "show",
    "ThreadSafeKVStorage"
]

if WITH_CUDA:
    __all__.extend(["infer_shape","supported_opset"])

import torchpipe.libipipe as _C
__all__.extend(["Event", '_C'])
