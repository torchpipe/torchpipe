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

import torchpipe
import torch
from torchvision import models
import torchvision
import os
import tempfile
from typing import Tuple, Optional
import torchpipe


class Arange(torch.nn.Module):
    def __init__(self):
        super(Arange, self).__init__()

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_k_seq_len: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        position_ids = torch.arange(
            0, q_len, dtype=torch.long).unsqueeze(0)  # .reshape((1, q_len))
        # tmp = torch.tensor((1,)).repeat(past_k_seq_len.squeeze(1)).size()[0]
        position_ids += past_k_seq_len.squeeze(1)

        # final_q_len = torch.clamp(final_q_len, max=final_q_len.item())
        return position_ids, past_k_seq_len


class TestUtilsModels:
    @classmethod
    def setup_class(self):

        tmpdir = tempfile.gettempdir()
        onnx_path = os.path.join(tmpdir, "arange.onnx")
        model = Arange().eval()

        input = (torch.randn((1, 2, 224)), torch.randint(12, 13, (1, 1)))

        dynamic_axes = {"hidden": {1: "seq_len"}, "output": {1: "seq_len"}}
        torch.onnx.export(
            model,
            input,
            onnx_path,
            opset_version=17,
            input_names=["hidden", "seq_len"],  # 输入名
            output_names=["output", "output2"],  # 输出名
            dynamic_axes=dynamic_axes,
        )
        print(f"saved onnx : {onnx_path} ")

        self.dict_args = {
            "model": onnx_path,
            "backend": "SyncTensor[TensorrtTensor]",
            "max": "1x20x224,1",
            "min": "1x1x224,1"
        }

        self.model = torchpipe.pipe(self.dict_args)

    def test_onnx_export_range(self):
        input = [torch.randn((1, 4, 224)),  torch.randint(17, 18, (1, 1))]
        print("input ", input[1])
        input_dict = {"data": input}
        self.model(input_dict)
        print(input_dict["result"][0])
        print(input_dict["result"][1].shape)
        assert (torch.equal(input_dict["result"][0].cpu(), torch.tensor(
            [[17, 18, 19, 20]],  dtype=torch.int32)))
        # assert (torch.equal(input_dict["result"][1].cpu(), torch.tensor([21])))
        # assert (input_dict["result"][1].shape[1] == 21)


if __name__ == "__main__":
    import time

    # time.sleep(10)
    a = TestUtilsModels()
    a.setup_class()
    a.test_onnx_export_range()
    # a.test_register_model()
