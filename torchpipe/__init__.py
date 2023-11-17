# Copyright 2021-2023 NetEase.
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

assert (
    __torch_version__ == torch.__version__
), "PyTorch version mismatch when compiling TorchPipe. Expected version {}, but got version {}.".format(
    __torch_version__, torch.__version__
)

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
    infer_shape,
    supported_opset,
)
from torchpipe.libipipe import Event
from torchpipe.libipipe import encrypt

# from torchpipe.utils import test, cpp_extension, patform, models
import torchpipe.utils
from .python_api import pipe

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

__all__ = [
    "Interpreter",
    "parse_toml",
    "TASK_DATA_KEY",
    "TASK_RESULT_KEY",
    "TASK_INFO_KEY",
    "pipe",
    "TASK_BOX_KEY",
    "TASK_NODE_NAME_KEY",
    "TASK_EVENT_KEY",
    "encrypt",
    # "any",
    "infer_shape",
    "supported_opset",
]


__all__.extend(["Event"])
