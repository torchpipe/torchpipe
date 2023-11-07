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


class MultiIdentities(torch.nn.Module):
    def __init__(self):
        super(MultiIdentities, self).__init__()
        self.identity = torch.nn.Identity()

    def forward(self, x, y):
        x = self.identity(x)
        return x, x+y

 
class TestBackend:
    @classmethod
    def setup_class(self):

        self.onnx_path =  os.path.join(
        tempfile.gettempdir(), f"tmp_identity_{random.random()}.onnx"
        )
        
        
        data = torch.rand((5, 224))
        identity_model = MultiIdentities().eval()
        torch.onnx.export(
        identity_model,
        (data, data),
        self.onnx_path,
        opset_version=13,
        do_constant_folding=True,
        input_names=["x", "y"],
        output_names=["output", "outputB"],
        dynamic_axes={
            "x": {0: "batch_size"},  # 批处理变量
            "y": {0: "batch_size"},  # 批处理变量
            "output": {0: "batch_size"},  # 批处理变量
            "outputB": {0: "batch_size"},
        },
        )
        print(f"{self.onnx_path} saved")
 
    def test_borrow(self):
        

        data = torch.rand((5, 224))

 
        ts = torchpipe.pipe({"a":{"model": self.onnx_path,"instance_num":3, 
                                  "instances_grp":"1;0,2",
                                  "backend": "Torch[TensorrtTensor]",
                                  "device_id":1, 'batching_timeout': '5',
                                  "max":"6,6;4,4;6,6",
                                  "min":"1,1"},
                             "b":{"model": "not_exists",
                                  "borrow_from":"a","active_instances_grp":"1",
                                   "backend": "Torch[TensorrtTensor]","device_id":1, 'batching_timeout': '5'}})
        input = {"data": [data, data],"node_name": "b"}
        ts(input)
        print(input["result"][1].shape, input["result"][1].device)
    
 
if __name__ == "__main__":
    pass

    a = TestBackend()

    a.setup_class()
    a.test_borrow()

    # a.test_batch(ts_schedule_pipe())
    # a.test_batch()

    # pytest.main([__file__])
