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

import io
from collections import OrderedDict
from typing import List, Tuple

import os
import pytest
import cv2
import numpy as np

import torch
import torchpipe
from torchpipe import pipe, TASK_DATA_KEY, TASK_RESULT_KEY, parse_toml
import tempfile


class IdentityModel(torch.nn.Module):
    def __init__(self):
        super(IdentityModel, self).__init__()
        self.identity = torch.nn.Identity()

    def forward(self, data):
        x = self.identity(data)
        return x


class TestBackend:
    @classmethod
    def setup_class(self):
        import tempfile
        tmpdir = tempfile.gettempdir()
        self.out_file = os.path.join(tmpdir, "py_max_identity.onnx")
        dummy_input = torch.randn(1, 3, 224, 224)

        model = IdentityModel().eval()
        torch.onnx.export(model, dummy_input, self.out_file,
                          opset_version=13,
                          do_constant_folding=True,
                          input_names=["input"], output_names=["output"],
                          dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}})

        self.model = torchpipe.pipe({"backend": "Torch[TensorrtTensor]", "model": self.out_file,
                                    "max": 3, "model::cache": self.out_file .replace(".onnx", ".trt")})

    def test_py_max(self):
        config = {"Interpreter::backend": "Max", "max": 4}
        model = pipe(config)
        assert (4 == model.max())
        assert (1 == model.min())

    def test_trt_max(self):
        self.model_trt = torchpipe.pipe(
            {"Interpreter::backend": "Torch[TensorrtTensor]", "model": self.out_file .replace(".onnx", ".trt")})
        assert 3 == self.model_trt.max()
        assert 1 == self.model_trt.min()

    def test_trt_max_different(self):
        # todo throw an exception
        self.model_trt = torchpipe.pipe(
            {"Interpreter::backend": "Torch[TensorrtTensor]", "max": 9, "model": self.out_file .replace(".onnx", ".trt")})
        assert 3 == self.model_trt.max()
        assert 1 == self.model_trt.min()
        del self.model_trt
        self.model_trt = None


if __name__ == "__main__":
    import time
    # time.sleep(5)
    a = TestBackend()
    a.setup_class()
    a.test_trt_max_different()
