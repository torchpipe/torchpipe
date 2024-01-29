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

from re import A
import torch
import io
import os

import torchpipe
from torchpipe import TASK_DATA_KEY, TASK_RESULT_KEY
import numpy as np
import tempfile
import pytest


class TestBackend:
    @classmethod
    def setup_class(self):

        self.pipe = torchpipe.pipe({"a": {"backend": "RuntimeError",
                                          'instance_num': 2, 'batching_timeout': '5'}, "b": {"backend": "RuntimeError"}})

        self.single_pipe = torchpipe.pipe({"a": {"backend": "RuntimeError",
                                                 'instance_num': 2, 'batching_timeout': '5'}, "b": {"backend": "RuntimeError"}})

        self.interpreter_pipe = torchpipe.pipe(
            {"Interpreter::backend": "C10Exception"})

    def test_infer(self):
        data = torch.rand((3, 1, 1))
        zz = np.ndarray((2, 22, 2))
        data = torch.from_numpy(zz)
        with pytest.raises(RuntimeError):
            self.pipe({"data": data, "node_name": "a"})
        with pytest.raises(RuntimeError):
            self.single_pipe({"data": 1, "node_name": "b"})

        with pytest.raises(IndexError):
            self.interpreter_pipe({"data": 1})

    def test_previous(self):
        p = torchpipe.pipe(
            {"0": {"next": "a"}, "a": {"next": "b"}, "b": {"map": "0[result:data]"}})
        input = {"data": 1}

        p(input)

    def test_map_no_data(self):
        p = torchpipe.pipe(
            {"a": {"next": "b"}, "b": {"map": "a[result:data1]"}})
        input = {"data": 1}
        with pytest.raises(RuntimeError):
            p(input)

    def test_nondirect_map(self):
        model = torchpipe.pipe("assets/nondirect_map.toml")
        data = torch.zeros(1, 3, 1, 224)
        input = {"data": data}
        model(input)
        assert (input["result"].shape[2] == 1)


if __name__ == "__main__":
    import time
    # time.sleep(5)
    a = TestBackend()
    a.setup_class()
    a.test_nondirect_map()
    # pytest.main([__file__])
