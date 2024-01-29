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
from torch import nn
from typing import Union, Tuple, List, Any
import torch


class Identity(torch.nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        self.identity = torch.nn.Identity()

    def forward(self, x):
        x = self.identity(x)
        return x


def get_onnx(model, batch_size):
    assert batch_size > 0 or batch_size == -1
    data_bchw = torch.rand((abs(batch_size), 3, 224, 224))
    out_file = f"./Identity_{batch_size}.onnx"
    if batch_size == -1:
        torch.onnx.export(
            model,
            data_bchw,
            out_file,
            opset_version=13,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={
                "input": {0: "batch_size"},  # variable lenght axes
                "output": {0: "batch_size"},
            },
        )
    else:
        torch.onnx.export(
            model,
            data_bchw,
            out_file,
            opset_version=13,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
        )


def get_dynamic_onnx(model):
    batch_size = -1
    data_bchw = torch.rand((abs(batch_size), 3, 224, 224))
    out_file = "./Identity_dynamic.onnx"
    torch.onnx.export(
        model,
        data_bchw,
        out_file,
        opset_version=13,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size", 2: "h", 3: "w"},  # variable lenght axes
            "output": {0: "batch_size", 2: "h", 3: "w"},
        },
    )


if __name__ == "__main__":
    model = Identity().eval()
    get_onnx(model, 1)
    get_onnx(model, 4)
    get_onnx(model, -1)
    get_dynamic_onnx(model)
    # onnxsim Identity_1.onnx Identity_1.onnx
    # time.sleep(10)
