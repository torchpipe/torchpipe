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

import torch
import io
import os
import pytest

import torchpipe
from torchpipe import TASK_RESULT_KEY, TASK_EVENT_KEY
import numpy as np
import tempfile
from timeit import default_timer as timer


class MultiIdentity(torch.nn.Module):
    def __init__(self):
        super(MultiIdentity, self).__init__()
        self.identity = torch.nn.Identity()

    def forward(self, data):
        x, y = data
        x = self.identity(x)
        return x, x+y


class TestBackend:
    @classmethod
    def setup_class(self):

        self.model = torchpipe.pipe("assets/PipelineV3.toml")
        # import time
        # time.sleep(10)
        # test stop filter with parallel tasks
        self.model_stop = torchpipe.pipe("assets/PipelineV3_stop.toml")
        # self.model_previous = torchpipe.pipe("assets/PipelineV3_previous.toml")

        self.single_map = torchpipe.pipe("assets/PipelineV3_single_map.toml")

    def test_except(self):
        data = torch.rand((3, 224, 282))
        # import time
        input = {"data": [data, data*2], "node_name": "jpg_decoder"}

        with pytest.raises(RuntimeError):
            self.model(input)  # 可并发调用
        # 使用 "result" 作为数据输出标识；当然，其他键值也可自定义写入

    def test_stop(self):
        data = torch.rand((3, 224, 282))

        input = {"data": [data, data*2], "node_name": "jpg_decoder"}
        self.model_stop(input)  # 可并发调用
        print(input.keys())
        # assert TASK_RESULT_KEY not in input.keys()

    def test_previous(self):
        data = torch.rand((3, 224, 224))
        input = {"data": [data], "node_name": "jpg_decoder"}
        with pytest.raises(RuntimeError):
            torchpipe.pipe("assets/PipelineV3_previous.toml")
            # model_previous(input)
            # print(input.keys())

    # def test_root_map(self):
    #     with pytest.raises(ValueError):
    #         torchpipe.pipe({"map":"a[b:c]", "Interpreter::backend":"PipelineV3"})
    #         # model_previous(input)
    def test_single_map(self):
        data = torch.rand((3, 224, 224))
        input = {"data": [data], "node_name": "jpg_decoder"}
        with pytest.raises(RuntimeError):
            self.single_map(input)
        print("finish test_single_map")

        # model_previous(input)
    def test_event(self):
        a = torchpipe.Event()
        # a.Wait()
        print(a.time_passed())
        assert a.time_passed() < 1 and a.time_passed() > 0

        data = torch.rand((3, 224, 282))
        # import time
        input = {"data": [data, data*2],
                 "node_name": "jpg_decoder", TASK_EVENT_KEY: a}
        self.model(input)  # 可并发调用
        with pytest.raises(RuntimeError):
            a.Wait()
        assert (input[TASK_EVENT_KEY] == a)

    def test_async_v0(self):
        event_pipe = torchpipe.pipe("assets/PipelineV3_event_sleep.toml")
        a = torchpipe.Event()

        input = {"data": 1, "node_name": "jpg_decoder", TASK_EVENT_KEY: a}
        print("start wait")
        start = timer()
        event_pipe(input)

        time_used = (timer() - start)*1000
        assert (time_used < 1)
        print(time_used)
        print(input.keys())
        assert TASK_RESULT_KEY not in input.keys()

        start = timer()
        a.Wait()
        time_used = (timer() - start)*1000
        print(time_used)
        # assert(time_used > 100)
        assert TASK_EVENT_KEY in input.keys()
        assert (input[TASK_EVENT_KEY] == a)

        b = torchpipe.Event()
        input_2 = {"data": 1, "node_name": "jpg_decoder", TASK_EVENT_KEY: b}
        print("start wait")
        event_pipe(input_2)
        b.Wait()

        assert input["result"] == 1
        input = {"data": 1, "node_name": "jpg_decoder"}
        event_pipe(input)
        assert input["result"] == 1

    def test_async(self):
        event_pipe = torchpipe.pipe({"backend": "Sleep", "Sleep::time": 100})
        a = torchpipe.Event()

        input = {"data": 1, "node_name": "jpg_decoder", TASK_EVENT_KEY: a}
        print("start wait")
        start = timer()
        event_pipe(input)

        time_used = (timer() - start)*1000
        assert (time_used < 1)
        print(time_used)
        print(input.keys())
        assert TASK_RESULT_KEY not in input.keys()

        start = timer()
        a.Wait()
        time_used = (timer() - start)*1000
        print(time_used)
        assert (time_used > 100)
        assert TASK_EVENT_KEY in input.keys()
        assert (input[TASK_EVENT_KEY] == a)

        b = torchpipe.Event()
        input_2 = {"data": 1, "node_name": "jpg_decoder", TASK_EVENT_KEY: b}
        print("start wait")
        event_pipe(input_2)
        b.Wait()

        assert input["result"] == 1
        input = {"data": 1, "node_name": "jpg_decoder"}
        event_pipe(input)
        assert input["result"] == 1

    def test_no_map(self):
        # torchpipe.pipe("./assets/PipelineV3_loss_map.toml")
        with pytest.raises(RuntimeError):
            torchpipe.pipe("./assets/PipelineV3_loss_map.toml")

    def test_gpu_cpu(self):
        p = torchpipe.pipe("./assets/PipelineV3_gpu_cpu.toml")
        with open("assets/damaged_jpeg/corrupt.jpg", "rb") as f:
            data = f.read()
        input = {"data": data, "node_name": "jpg_decoder"}
        p(input)
        assert (input["result"].shape[1] == 3)

    def test_next(self):
        p = torchpipe.pipe({"decoder": {"backend": "SyncTensor[DecodeTensor]", "next": "resizer"},
                            "resizer": {"backend": "SyncTensor[Tensor2Mat]"}})
        with open("assets/damaged_jpeg/corrupt.jpg", "rb") as f:
            data = f.read()
        input = {"data": data, "node_name": "decoder"}
        p(input)
        assert "result" not in input.keys()

        with open("assets/norm_jpg/dog.jpg", "rb") as f:
            data = f.read()
        input = {"data": data, "node_name": "decoder"}
        p(input)

        assert (input["result"].shape[1] == 768)


if __name__ == "__main__":
    import time
    # time.sleep(10)
    a = TestBackend()
    a.setup_class()
    a.test_next()
    # a.test_single_map()
    # for i in range(1000):
    #     a.test_async_v0()
    # exit()
    # a.test_async()
    # a.test_event()
    # a.test_no_map()

    # a.test_stop()
    # a.test_stop()
    # a.test_previous()

    # pytest.main([__file__])
