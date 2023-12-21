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


class TestBackend:
    @classmethod
    def setup_class(self):
        torch.manual_seed(123)
        self.nchw = torch.randn((222, 333, 3)).cuda().permute(
            2, 0, 1).unsqueeze(0)
        self.hwc = torch.randn((1, 3, 222, 333)).cuda().permute(
            0, 2, 3, 1).squeeze(0)

        self.cpu_hwc = torch.randn(
            (1, 3, 222, 333)).permute(0, 2, 3, 1).squeeze(0)

    def test_cvt(self):
        config = {'cvt':
                  {'backend': "S[CvtColorTensor,SyncTensor]", "color": "rgb"}}

        model = pipe(config)

        input_dict = {TASK_DATA_KEY: self.nchw, "color": "bgr"}
        model(input_dict)
        assert (torch.equal(input_dict[TASK_RESULT_KEY], torch.flip(
            self.nchw, dims=(1,)).clone()))
        assert (not input_dict[TASK_RESULT_KEY].is_contiguous())

        input_dict = {TASK_DATA_KEY: self.nchw, "color": "rgb"}
        model(input_dict)
        assert (torch.equal(input_dict[TASK_RESULT_KEY], self.nchw.clone()))

        input_dict = {TASK_DATA_KEY: self.hwc, "color": "bgr"}
        model(input_dict)
        assert (torch.equal(input_dict[TASK_RESULT_KEY], torch.flip(
            self.hwc, dims=(2,)).clone()))

        input_dict = {TASK_DATA_KEY: self.cpu_hwc, "color": "bgr"}
        model(input_dict)
        assert (torch.equal(input_dict[TASK_RESULT_KEY], torch.flip(
            self.cpu_hwc, dims=(2,)).clone()))

    def test_nvcvt(self):
        config = {'cvt':
                  {'backend': "S[Tensor2NvTensor,CvtColorNvTensor,NvTensor2Tensor,SyncTensor]", "color": "rgb"}}

        model = pipe(config)

        input_dict = {TASK_DATA_KEY: self.nchw, "color": "bgr"}
        model(input_dict)
        assert(input_dict[TASK_RESULT_KEY].stride() == (999,3,1))
        assert (torch.equal(input_dict[TASK_RESULT_KEY], torch.flip(
            self.nchw, dims=(1,)).permute((0,2,3,1)).squeeze(0)))
        
        assert ( input_dict[TASK_RESULT_KEY].is_contiguous())


        input_dict = {TASK_DATA_KEY: self.nchw.to(torch.uint8), "color": "bgr"}
        model(input_dict)
        assert (torch.equal(input_dict[TASK_RESULT_KEY], torch.flip(
            self.nchw.to(torch.uint8), dims=(1,)).permute((0,2,3,1)).squeeze(0)))


        input_dict = {TASK_DATA_KEY: self.nchw, "color": "rgb"}
        model(input_dict)
        assert (torch.equal(input_dict[TASK_RESULT_KEY], self.nchw.permute((0,2,3,1)).squeeze(0)))

        input_dict = {TASK_DATA_KEY: self.hwc, "color": "bgr"}
        model(input_dict)
        assert (torch.equal(input_dict[TASK_RESULT_KEY], torch.flip(
            self.hwc, dims=(2,))))

        input_dict = {TASK_DATA_KEY: self.hwc, "color": "rgb"}
        model(input_dict)
        assert (torch.equal(input_dict[TASK_RESULT_KEY], self.hwc))
        assert(input_dict[TASK_RESULT_KEY].is_contiguous())

        input_dict = {TASK_DATA_KEY: self.cpu_hwc, "color": "bgr"}
        with pytest.raises(RuntimeError):
            model(input_dict)
 

if __name__ == "__main__":
    import time
    # time.sleep(5)
    a = TestBackend()
    a.setup_class()
    a.test_nvcvt()
