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
        torch.manual_seed(123)

        jpg_path = "assets/encode_jpeg/grace_hopper_517x606.jpg"
        self.gray = "./assets/image/gray.jpg"

    def test_gray(self):
        Torch_Loops = ["Torch[Torch[Identity]]",
                       "TensorSync[TensorSync]", "SyncTensor[SyncTensor]"]

        config = {'jpg_decoder':
                  {'backend': f"Torch[Torch[S[C10Exception]]]"}}
        with pytest.raises(ValueError):
            model = pipe(config)

    def test_gray2(self):
        Torch_Loops = ["Torch[Identity]",
                       "TensorSync[TensorSync]", "SyncTensor[SyncTensor]"]

        config = {'jpg_decoder':
                  {'backend': f"Torch[S[{Torch_Loops[0]},{Torch_Loops[1]},S[{Torch_Loops[2]}]]]"}}
        model = pipe(config)

        input_dict = {TASK_DATA_KEY: 1}

        model(input_dict)

        assert (input_dict["result"] == 1)

    def test_sts(self):
        config = {'jpg_decoder':
                  {'backend': f"S[Torch[S]]"}}
        with pytest.raises(RuntimeError):
            model = pipe(config)

    def test_tst(self):
        config = {'jpg_decoder':
                  {'backend': f"SyncTensor[S[Identity]]"}}
        model = pipe(config)

    def test_sts12(self):
        Torch_Loops = ["Torch[Identity]",
                       "TensorSync[TensorSync]", "SyncTensor[SyncTensor]"]

        config = {'jpg_decoder':
                  {'backend': f"Torch[S[{Torch_Loops[0]},{Torch_Loops[1]},S[C10Exception]]]"}}
        model = pipe(config)

        input_dict = {TASK_DATA_KEY: 1}
        with pytest.raises(RuntimeError):
            model(input_dict)
        assert 'result' not in input_dict.keys()

    def test_tstc10(self):
        config = {'jpg_decoder':
                  {'backend': f"Torch[S[Torch[Identity],Torch[C10Exception]]]"}}
        model = pipe(config)
        input_dict = {TASK_DATA_KEY: 1}
        with pytest.raises(RuntimeError):
            model(input_dict)
        assert 'result' not in input_dict.keys()


if __name__ == "__main__":
    import time
    # time.sleep(5)
    a = TestBackend()
    a.setup_class()
    a.test_tstc10()
