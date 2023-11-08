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
        self.out_file = os.path.join(tmpdir, "range.onnx")
        dummy_input = torch.randn(1, 3, 224, 224)

        model = IdentityModel().eval()
        torch.onnx.export(model, dummy_input, self.out_file,
                          opset_version=13,
                          do_constant_folding=True,
                          input_names=["input"], output_names=["output"],
                          dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}})

 


 
    def test_trt_max(self):
        model_trt = torchpipe.pipe(
            {"Interpreter::backend": "Range[S[TensorrtTensor,SyncTensor]]",
             "range":"8,19","max": 9,"min":2, "model": self.out_file})
        
        assert 19 == model_trt.max()
        assert 8 == model_trt.min()

        with pytest.raises(RuntimeError):
            model_trt = torchpipe.pipe(
                {"Interpreter::backend": "Range[S[TensorrtTensor,SyncTensor]]",
                "range":"8,9","max": 8,"min":8, "model": self.out_file})
        
    def test_force_range(self):
        model_trt = torchpipe.pipe(
            {"Interpreter::backend": "S[TensorrtTensor,SyncTensor]",
             "force_range":"1,1","max": 10,"min":10, "model": self.out_file})
        
        print(model_trt.max())
        assert 1 == model_trt.max()
        assert 1 == model_trt.min()


        model_trt = torchpipe.pipe(
                {"backend": "S[TensorrtTensor,SyncTensor]",
                "force_range":"1,1","max": 8,"min":8, "model": self.out_file})
        
        dummy_input = torch.randn(8, 3, 224, 224)
        input = {"data":dummy_input}
        model_trt(input)
        assert input["result"].shape[0] == 8

        with pytest.raises(RuntimeError):
            dummy_input = torch.randn(2, 3, 224, 224)
            input = {"data":dummy_input}
            model_trt(input)
        
        
 
if __name__ == "__main__":
    import time
    # time.sleep(5)
    a = TestBackend()
    a.setup_class()
    a.test_force_range()
