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
# In environments without onnxruntime we prefer to
# invoke all tests in the repo and have this one skipped rather than fail.
# onnxruntime = pytest.importorskip("onnxruntime")
import torch
import torchpipe
from torchpipe import pipe, TASK_DATA_KEY, TASK_RESULT_KEY, parse_toml
import tempfile


class TestBackend:
    @classmethod
    def setup_class(self):
        torch.manual_seed(123)
        import torchvision.models as models
        resnet18 = models.resnet18(pretrained=True).eval().cuda()

        jpg_path = "assets/encode_jpeg/grace_hopper_517x606.jpg"
        self.gray = "./assets/image/gray.jpg"

        # self.img = cv2.imread(jpg_path)

    def decode_run(self, input_dict, target, model):
        model(input_dict)

        assert (input_dict["color"] in [b"rgb", b"bgr"])

        assert (input_dict["color"] == b"rgb")
        input_dict[TASK_RESULT_KEY] = input_dict[TASK_RESULT_KEY][:,
                                                                  [2, 1, 0], :, :]
        assert (len(input_dict[TASK_RESULT_KEY].squeeze(0)) == 3)
        input_dict[TASK_RESULT_KEY] = input_dict[TASK_RESULT_KEY].squeeze(
            0).permute(1, 2, 0)
        assert (input_dict[TASK_RESULT_KEY].shape == (274, 442, 3))
        if target:
            z = (target - input_dict[TASK_RESULT_KEY].float())
            rel = torch.mean(z).item()

            assert (abs(rel) < 1)

    def test_gray(self):

        config = {'jpg_decoder':
                  {'backend': "Sequential[DecodeTensor,SyncTensor]"}}

        model = pipe(config)

        with open(self.gray, "rb") as f:
            raw_jpg = f.read()

        input_dict = {TASK_DATA_KEY: raw_jpg}

        self.decode_run(input_dict, None, model)


if __name__ == "__main__":
    import time
    # time.sleep(10)
    a = TestBackend()
    a.setup_class()
    a.test_gray()
