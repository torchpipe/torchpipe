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


class TestBackend:
    @classmethod
    def setup_class(self):
        self.inputs = [
            torch.rand(1, 3, 224, 224).cuda(0),
            torch.rand(1, 3, 224, 224),
            [torch.rand(1, 3, 224, 224).cuda(0), torch.rand(1, 3, 224, 224)],
            torch.rand(1, 3, 224, 224).cuda(0).half(),
            torch.empty((1, 3, 224, 224),
                        dtype=torch.float32, pin_memory=True),
        ]

    # 只有一个设备时跳过
    @pytest.mark.skipif(torch.cuda.device_count() < 2, reason="only one device")
    def test_switch_device(self):
        config = {"backend": f"Torch[ Identity]", "device_id": "1"}
        CUDA_VISIBLE_DEVICES = os.environ.get("CUDA_VISIBLE_DEVICES", None)
        if not CUDA_VISIBLE_DEVICES is None:
            len_cuda = len(CUDA_VISIBLE_DEVICES.split(","))
            if len_cuda < 2:
                pytest.skip("len(CUDA_VISIBLE_DEVICES) < 2")

        model = pipe(config)
        for input0 in self.inputs:
            if isinstance(input0, torch.Tensor):
                input0 = input0.cuda(0)
            elif isinstance(input0, list):
                for i in range(len(input0)):
                    input0[i] = input0[i].cuda(0)
            input_dict = {TASK_DATA_KEY: input0}
            model(input_dict)

            if isinstance(input0, torch.Tensor):
                assert input_dict["result"].device == torch.device("cuda:1")
                assert torch.equal(input0.cuda(1), input_dict["result"])
            elif isinstance(input0, list):
                for input_data, result_data in zip(input0, input_dict["result"]):
                    assert result_data.device == torch.device("cuda:1")
                    assert torch.equal(input_data.cuda(1), result_data)


if __name__ == "__main__":
    import time

    # time.sleep(5)
    a = TestBackend()
    a.setup_class()
    a.test_switch_device()
