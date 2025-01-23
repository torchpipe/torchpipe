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

import torchpipe
import torch
# from torchvision import models
# import torchvision
import os
import tempfile
from typing import Tuple, Optional
import torchpipe


class Slice(torch.nn.Module):
    def __init__(self):
        super(Slice, self).__init__()
        self.max_seq_len_cached = 100
        self.cos_cached = torch.zeros((1, self.max_seq_len_cached))

    def forward(self, x, seq_len=None):

        return (
            self.cos_cached[:,:seq_len[0].view(())].to(dtype=x.dtype),
        )


class TestUtilsModels:
    @classmethod
    def setup_class(self):
        
        return
        tmpdir = tempfile.gettempdir()
        onnx_path = os.path.join(tmpdir, "arange.onnx")
        model = Slice().eval()


        input_shape = (1, 12, 100, 64)
        input_names = ["x", "seq_len"]
        output_names = ["cos_cached"]

        input = (torch.randn(input_shape), torch.tensor([50]).unsqueeze(0))
        path_onnx = onnx_path
        torch.onnx.export(model, input, path_onnx, opset_version=11,
                        input_names=input_names, output_names=output_names,
                        dynamic_axes={"x": {0: "batch_size", 2: "sequence_length"},
                                        "cos_cached": {0: "batch_size", 1: "sequence_length"}})
        os.system(f"onnxsim {onnx_path} {onnx_path}")
        print(f"saved onnx : {onnx_path} ")

        self.dict_args = {
            "model": onnx_path,
            "backend": "SyncTensor[TensorrtTensor]",
            "max": "1x1",
            "min": "1x1"
        }

        self.model = torchpipe.pipe(self.dict_args)

    def test_slice(self):
        return
        input = torch.randint(50, 51, (1, 1))
        input_dict = {"data": input}
        self.model(input_dict)
        print(input_dict["result"][0].shape)
        # assert (torch.equal(input_dict["result"][0].cpu(), torch.tensor(
        #     [[17, 18, 19, 20]],  dtype=torch.int32)))
        # assert (torch.equal(input_dict["result"][1].cpu(), torch.tensor([21])))
        # assert (input_dict["result"][1].shape[1] == 21)
    
         
 

if __name__ == "__main__":
    import time

    # time.sleep(10)
    a = TestUtilsModels()
    a.setup_class()
    a.test_slice()
 