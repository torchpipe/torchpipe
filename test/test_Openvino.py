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
import tempfile


def schedule_pipe(identity_model=None):
    if identity_model is None:
        identity_model = Identity1().eval()

    data_bchw = torch.rand((1, 3, 224, 224))

    onnx_path = os.path.join(tempfile.gettempdir(), "ov_identity1.onnx")
    print("export: ", onnx_path)
    # torch.onnx.export(
    #     identity_model,
    #     data_bchw,
    #     onnx_path,
    #     opset_version=13,
    #     do_constant_folding=True,
    #     input_names=["input"],
    #     output_names=["output"],
    #     dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    # )
    torch.onnx.export(
        identity_model,
        data_bchw,
        onnx_path,
        opset_version=13,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"]
        )
    ov_model = torchpipe.pipe(
        {
            "model": onnx_path,
            "backend": "S[OpenvinoMat]",
            "instance_num": 1,
            "batching_timeout": "5",
        }
    )
    return ov_model


class Identity1(torch.nn.Module):
    def __init__(self):
        super(Identity1, self).__init__()
        self.identity = torch.nn.Identity()
         
    def forward(self, data):
        return self.identity(data)

@pytest.mark.skipif(not torchpipe.libipipe.WITH_OPENVINO, reason="WITH_OPENVINO is False")
class TestBackend:
    @classmethod
    def setup_class(self):
        
        self.identity_model = schedule_pipe()

        # self.data_bchw = torch.rand((1, 3, 224, 282))
    def test_1(self):
        # identity_model = schedule_pipe(Identity2().eval())
        data = torch.randn((5, 3, 224, 224))
        assert(data.is_contiguous())
        assert(data[2:,...].is_contiguous())    
        # input = [{"data": i.unsqueeze(0)} for i in data[2:,...]]
        
        data = torch.randn((224, 224, 3)).numpy()#.astype(dtype=np.uint8)
        input= {"data": data}
        self.identity_model(input)
        result = input["result"]
        # compare data = result in numpy
        print(result.shape)
        assert(np.array_equal(data, result))
        
        # assert(input["result"].is_contiguous())

        # assert(torch.equal(input[0]["result"].squeeze(), data[2]))
        # assert(torch.equal(input[2]["result"].squeeze(), data[4]))
 
if __name__ == "__main__":
    pass

    a = TestBackend()

    a.setup_class()
    # import time
    # time.sleep(5)
    a.test_1()
    # a.test_infer()