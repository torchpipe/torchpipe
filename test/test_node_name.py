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
# In environments without onnxruntime we prefer to
# invoke all tests in the repo and have this one skipped rather than fail.
# onnxruntime = pytest.importorskip("onnxruntime")
import torch
import torchpipe
from torchpipe import pipe, TASK_DATA_KEY, TASK_RESULT_KEY, parse_toml
import tempfile


class TestTensor:
    @classmethod
    def setup_class(self):
        pass

    @pytest.mark.parametrize('backend', ['SyncTensor[ResizeTensor]', 'SyncTensor[cvtColorTensor]', 'Sequential[ResizeTensor,cvtColorTensor,SyncTensor]', 'Sequential[cvtColorTensor,ResizeTensor,SyncTensor]'])
    def test_input(self, backend):
        config = {"dd": {'Interpreter::backend': f"{backend}",
                         "color": "rgb",
                         "resize_h": "224",
                         "resize_w": "224"}}

        model = pipe(config)

        jpg_path = "assets/encode_jpeg/grace_hopper_517x606.jpg"
        img = cv2.imread(jpg_path)
        data = torch.from_numpy(img).float()
        input_dict = {TASK_DATA_KEY: data, 'color': "bgr", "node_name": "zz"}

        model(input_dict)
        assert len(input_dict[TASK_RESULT_KEY].shape) == 3
        assert (input_dict[TASK_RESULT_KEY].shape[2] == 3)

        data = torch.from_numpy(img).permute((2, 0, 1)).unsqueeze(0)
        input_dict = {TASK_DATA_KEY: data, 'color': "bgr"}
        model(input_dict)
        assert len(input_dict[TASK_RESULT_KEY].shape) == 4
        assert (input_dict[TASK_RESULT_KEY].shape[1] == 3)

    def test_multiple_nodes(self):
        backend = "Identity"
        config = {"dd": {'Interpreter::backend': f"{backend}",
                         "color": "rgb",
                         "resize_h": "224",
                         "resize_w": "224"},
                  "cc": {'Interpreter::backend': f"{backend}",
                         "color": "rgb",
                         "resize_h": "224",
                         "resize_w": "224"}}

        model = pipe(config)

        jpg_path = "assets/encode_jpeg/grace_hopper_517x606.jpg"
        img = cv2.imread(jpg_path)
        data = torch.from_numpy(img).float()
        input_dict = {TASK_DATA_KEY: data, 'color': "bgr", "node_name": "zz"}
        with pytest.raises(IndexError):
            model(input_dict)
        # assert len(input_dict[TASK_RESULT_KEY].shape) == 4
        # assert(input_dict[TASK_RESULT_KEY].shape[1]==3)

        data = torch.from_numpy(img).permute((2, 0, 1)).unsqueeze(0)
        input_dict = {TASK_DATA_KEY: data, 'color': "bgr"}
        with pytest.raises(IndexError):
            model(input_dict)
        # assert len(input_dict[TASK_RESULT_KEY].shape) == 4
        # assert(input_dict[TASK_RESULT_KEY].shape[1]==3)


if __name__ == "__main__":
    import time
    # time.sleep(10)
    a = TestTensor()
    a.setup_class()
