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

    data_bchw = torch.rand((1, 3, 223, 224))

    onnx_path = os.path.join(tempfile.gettempdir(), "ov_identity1.onnx")
    print("export: ", onnx_path)

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

def multiple_outputs_pipeline(onnx_path=None, num_input=1, num_output=2):
    if onnx_path is None:
        from onnx_generator import generate_identity_onnx
        onnx_path = generate_identity_onnx(num_input=num_input, num_output=num_output)
        
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
        return self.identity(data) + 0

@pytest.mark.skipif(not torchpipe.libipipe.WITH_OPENVINO, reason="WITH_OPENVINO is False")
class TestBackend:
    @classmethod
    def setup_class(self):
        
        self.identity_model = schedule_pipe()
        self.identity_multiple_outputs = multiple_outputs_pipeline(num_input=1, num_output=2)

        # self.data_bchw = torch.rand((1, 3, 224, 282))
    def test_1(self):
        # identity_model = schedule_pipe(Identity2().eval())
        data = torch.randn((5, 3, 223, 224))
        assert(data.is_contiguous())
        assert(data[2:,...].is_contiguous())    
        # input = [{"data": i.unsqueeze(0)} for i in data[2:,...]]
        
        data = torch.randn((223, 224, 3)).numpy()#.astype(dtype=np.uint8)
        input= {"data": data}
        self.identity_model(input)
        result = input["result"]

        # result=result.reshape(224, 224, 3)
        # compare data = result in numpy
        print(result.shape, data[0,0,:])
        assert(np.array_equal(data, result))

    def test_instance(self):
        return
        # identity_model = schedule_pipe(Identity2().eval())
        data = torch.randn((5, 3, 223, 224))
        assert(data.is_contiguous())
        assert(data[2:,...].is_contiguous())    
        # input = [{"data": i.unsqueeze(0)} for i in data[2:,...]]
        
        data = torch.randn((223, 224, 3)).numpy()#.astype(dtype=np.uint8)
        input= {"data": data}
        self.identity_model(input)
        result = input["result"]

        # result=result.reshape(224, 224, 3)
        # compare data = result in numpy
        print(result.shape, data[0,0,:])
        assert(np.array_equal(data, result))
        
    def test_multiple_outputs(self):
        
        data = torch.randn((223, 224, 3)).numpy()#.astype(dtype=np.uint8)
        input= {"data": data}
        self.identity_multiple_outputs(input)
        result = input["result"]

        assert(len(result) == 2)
        print(result[0].shape)
        assert(np.array_equal(data, result[0]))
        assert(np.array_equal(data+1, result[1]))

 
 
    def test_various_outputs(self):
        from onnx_generator import generate_various_type_outputs
        onnx_path = generate_various_type_outputs()
        
        ov_model = torchpipe.pipe(
            {
                "model": onnx_path,
                "backend": "S[OpenvinoMat]",
                "instance_num": 10,
            }
        )
        with pytest.raises(RuntimeError):
            torchpipe.pipe(
                {
                    "model": onnx_path,
                    "backend": "S[OpenvinoMat]",
                    "instance_num": 110,
                }
            )
 
        data = torch.randn((223, 224, 3)).numpy()#.astype(dtype=np.uint8)
        input= {"data": data}
        ov_model(input)
        result = input["result"]

        assert(len(result) == 4)
        print([result[i].shape for i in range(4)])
        assert(np.array_equal(data, result[0]))
        assert(np.array_equal(data+1, result[1]))

        mean = data.transpose(2,0,1).mean(axis=-1).squeeze()
        # a和result[3]的最大相对误差
        assert(np.allclose(mean, result[3], rtol=1e-3, atol=1e-5))
        
        # gap
        gap = torch.nn.AdaptiveAvgPool2d((1, 1))(torch.tensor(data).permute(2,0,1).unsqueeze(0)).squeeze()
        gap = gap.unsqueeze(0)
        assert(np.allclose(gap, result[2], rtol=1e-3, atol=1e-5))

if __name__ == "__main__":
    # import time
    # time.sleep(5)

    a = TestBackend()

    a.setup_class()
    
    a.test_various_outputs()
    # a.test_infer()