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
import io
import os

import torchpipe
import numpy as np
import tempfile


import pytest

from base_config import pipelines, schedules
import random


@pytest.fixture(scope="class", params=list(zip(pipelines, schedules)))
def schedule_pipe(request):
    identity_model = Identity().eval()

    data_bn = torch.rand((1, 1000))

    onnx_path = os.path.join(
        tempfile.gettempdir(), f"tmp_identity_{random.random()}.onnx"
    )
    print("export: ", onnx_path)
    torch.onnx.export(
        identity_model,
        data_bn,
        onnx_path,
        opset_version=13,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"},
                      "output": {0: "batch_size"}},  # 批处理变量
    )
    pipeline, schedule = request.param
    tensorrt = torchpipe.pipe(
        {
            "model": onnx_path,
            "backend": "SyncTensor[TensorrtTensor]",
            "instance_num": 2,
            "batching_timeout": "5",
            "max": "4",
            "Interpreter::backend": pipeline,
            pipeline + "::backend": schedule,
            "postprocessor": "SoftmaxMax",
        }
    )
    return tensorrt


class Identity(torch.nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        self.identity = torch.nn.Identity()

    def forward(self, data):
        return self.identity(data)


class TestBackend:
    @classmethod
    def setup_class(self):
        pass

    def test_infer(self, schedule_pipe):
        data = torch.rand((1, 1000))
        max_value, max_index = torch.max(torch.softmax(data, dim=1), dim=1)
        print(max_value, max_index)
        input = {"data": data}
        schedule_pipe(input)  # 可并发调用
        assert int(input["result"][0]) == max_index
        assert abs((input["result"][1]) - max_value) / max_value < 0.01

    def test_batch_infer(self, schedule_pipe):
        data = torch.rand((1, 1000))
        max_value, max_index = torch.max(torch.softmax(data, dim=1), dim=1)
        print(max_value, max_index)
        input = {"data": data}
        inputs = [input] * 4
        schedule_pipe(inputs)  # 可并发调用

        input = inputs[3]
        assert int(input["result"][0]) == max_index
        assert abs((input["result"][1]) - max_value) / max_value < 0.01


if __name__ == "__main__":
    pass

    a = TestBackend()

    a.setup_class()
    # a.test_batch()

    # pytest.main([__file__])
