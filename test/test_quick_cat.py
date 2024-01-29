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
import io
import os

import torchpipe
import numpy as np
import tempfile

import random
import pytest

from base_config import pipelines, schedules


def schedule_pipe_multiple_inputs():
    identity_model = MultiIdentity().eval()

    data_bchw = torch.rand((1, 3, 224, 224))

    onnx_path = os.path.join(
        tempfile.gettempdir(), f"tmp_identity_{random.random()}.onnx"
    )
    print("export: ", onnx_path)
    torch.onnx.export(
        identity_model,
        [data_bchw, data_bchw],
        onnx_path,
        opset_version=13,
        do_constant_folding=True,
        input_names=["input", "inputB"],
        output_names=["output", "outputB"],
        dynamic_axes={
            "input": {0: "batch_size"},  # 批处理变量
            "inputB": {0: "batch_size"},  # 批处理变量
            "output": {0: "batch_size"},  # 批处理变量
            "outputB": {0: "batch_size"},
        },
    )
    tensorrt = torchpipe.pipe(
        {
            "model": onnx_path,
            "backend": "SyncTensor[TensorrtTensor]",
            "instance_num": 2,
            "batching_timeout": "5",
            "max": "4",
          }
    )
    return tensorrt


def schedule_pipe(identity_model=None):
    if identity_model is None:
        identity_model = Identity1().eval()

    data_bchw = torch.rand((1, 3, 224, 224))

    onnx_path = os.path.join(tempfile.gettempdir(), "cat_identity1.onnx")
    print("export: ", onnx_path)
    torch.onnx.export(
        identity_model,
        data_bchw,
        onnx_path,
        opset_version=13,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )
    tensorrt = torchpipe.pipe(
        {
            "model": onnx_path,
            "backend": "SyncTensor[TensorrtTensor]",
            "instance_num": 1,
            "batching_timeout": "5",
            "max": "4",
            "precision":"fp32"
        }
    )
    return tensorrt


class Identity1(torch.nn.Module):
    def __init__(self):
        super(Identity1, self).__init__()
        self.identity = torch.nn.Identity()
         
    def forward(self, data):
        return self.identity(data)


class MultiIdentity(torch.nn.Module):
    def __init__(self):
        super(MultiIdentity, self).__init__()
        self.identity = torch.nn.Identity()

    def forward(self, data):
        x, y = data
        x = self.identity(x)
        return x, x + y


class TestBackend:
    @classmethod
    def setup_class(self):
        
        self.identity_model = schedule_pipe()
        self.identity_model_m  =schedule_pipe_multiple_inputs()
        # self.data_bchw = torch.rand((1, 3, 224, 282))

    def test_1(self):
        # identity_model = schedule_pipe(Identity2().eval())
        data = torch.randn((5, 3, 224, 224)).cuda()
        assert(data.is_contiguous())
        assert(data[2:,...].is_contiguous())    
        input = [{"data": i.unsqueeze(0)} for i in data[2:,...]]
        self.identity_model(input)
        assert(input[0]["result"].is_contiguous())

        assert(torch.equal(input[0]["result"].squeeze(), data[2]))
        assert(torch.equal(input[2]["result"].squeeze(), data[4]))
    def test_2(self):
        data = torch.randn((5, 3, 224, 224)).cuda()
        data2 = torch.randn((6, 3, 224, 224)).cuda()
        assert(data.is_contiguous())
        assert(data[2:,...].is_contiguous())    
        input = [{"data": [i.unsqueeze(0), j.unsqueeze(0)]} for i, j in zip(data[2:,...], data2[3:,...])]
        self.identity_model_m(input)
        assert(input[0]["result"][1].is_contiguous())

        assert(torch.equal(input[0]["result"][1].squeeze(), data[2]+data2[3]))
        assert(torch.equal(input[2]["result"][1].squeeze(), data[4]+data2[5]))
        assert(torch.equal(input[2]["result"][0].squeeze(), data[4]))
 

if __name__ == "__main__":
    pass

    a = TestBackend()

    a.setup_class()
    # import time
    # time.sleep(5)
    a.test_2()
    # a.test_infer()

    # pytest.main([__file__])
