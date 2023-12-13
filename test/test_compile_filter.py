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
import torchpipe as tp
from torchpipe import pipe, TASK_DATA_KEY, TASK_RESULT_KEY, parse_toml
import tempfile


 
class TestBackend:
    @classmethod
    def setup_class(self):
        tp.utils.cpp_extension.load_filter(
            name = 'Skip', 
            sources='status forward(dict data){return status::Skip;}',
            sources_header="")


         
    def test_force_range(self):
        tp.utils.cpp_extension.load_backend(
            name = 'identity', 
            sources='void forward(dict data){(*data)["result"] = (*data)["data"];}',
            sources_header="")
        model = tp.pipe({"backend":'identity'})
        input = {"data":2}
        model(input)
        assert input["result"] == 2
        
        
if __name__ == "__main__":
    import time
    # time.sleep(5)
    a = TestBackend()
    a.setup_class()
    a.test_force_range()
