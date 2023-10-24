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

import torchpipe
import numpy as np
import tempfile


import pytest

from base_config import pipelines, schedules


def schedule_pipe(identity_model=None):
    if identity_model is None:
        identity_model = Identity1().eval()

    data_bchw = torch.rand((1, 3, 224, 224))

    onnx_path = os.path.join(tempfile.gettempdir(), "tmp_identity1.onnx")
    print("export: ", onnx_path)
    torch.onnx.export(
        identity_model,
        data_bchw,
        onnx_path,
        opset_version=13,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )
    tensorrt = torchpipe.pipe(
        {
            "model": onnx_path,
            "backend": "SyncTensor[TensorrtTensor]",
            "instance_num": 2,
            "batching_timeout": "5",
            "max": "4",
        }
    )
    return tensorrt


class Identity1(torch.nn.Module):
    def __init__(self):
        super(Identity1, self).__init__()
        self.identity = torch.nn.Identity()
        self.matmul = torch.nn.Linear(150528, 1)
        self.matmul2 = torch.nn.Linear(150528 // 3, 1)

    def forward(self, data):
        final = int(data.numel() // data.shape[0])
        return data.view(3, final // 3)

        x = data.view(1, -1)
        # x=self.matmul(x)
        # x=x.expand((1,150528)).clone()

        x += torch.ones((1, 1))
        x = data.reshape(3, -1)
        # x=self.matmul2(x)
        # x=x.expand((3,150528//3)).clone()
        x += torch.ones((3, 1)) * 2
        x = x.reshape(-1, final)
        return x


class Identity2(torch.nn.Module):
    def __init__(self):
        super(Identity2, self).__init__()

        self.avg = torch.nn.AdaptiveAvgPool1d((1, 1))

    def forward(self, data):
        return (data).view(1, -1)


class TestBackend:
    @classmethod
    def setup_class(self):
        pass
        # self.identity_model = Identity1().eval()

        # self.data_bchw = torch.rand((1, 3, 224, 282))

    def test_1(self):
        # identity_model = schedule_pipe(Identity2().eval())
        # data = torch.rand((1, 3, 224, 224))
        # input = {"data": data}
        with pytest.raises(ValueError):
            identity_model = schedule_pipe(Identity2().eval())

    # def test_infer(self):
    #     schedule_pipe()(input)  # 可并发调用
    #     data = torch.rand((1, 3, 224, 224))
    #     input = {"data": data}
    #     with pytest.raises(RuntimeError):
    #         schedule_pipe()(input)  # 可并发调用
    # 使用 "result" 作为数据输出标识；当然，其他键值也可自定义写入
    # 失败则此键值一定不存在，即使输入时已经存在。
    # assert(torch.allclose(3*input["result"][0], input["result"][1]))

    # def test_batch(self, schedule_pipe):
    #     data = np.random.randint(0, 255, (3, 224, 282), dtype=np.uint8)
    #     data = torch.from_numpy(data)
    #     inputs = []
    #     for i in range(5):
    #         inputs.append({"data": [data, data]})
    #     # for i in range(7):
    #     schedule_pipe(inputs)
    #     # 失败则此键值一定不存在，即使输入时已经存在。
    #     assert(torch.allclose(inputs[0]["result"]
    #            [0]*2, inputs[4]["result"][1]))

    # def test_batch_float(self, schedule_pipe):
    #     data = np.random.randint(0, 255, (3, 224, 282), dtype=np.uint8)
    #     # note that uint8 is lgt 255. Here we use float
    #     data = torch.from_numpy(data).float()
    #     from concurrent.futures import ThreadPoolExecutor
    #     with ThreadPoolExecutor(max_workers=8) as pool:
    #         futures = []
    #         inputA = {"data": [data.clone(), 2*data]}

    #         inputs_all = []
    #         future = pool.submit(schedule_pipe, inputA)
    #         # futures.append(future)
    #         inputs_all.append(inputA)
    #         for i in range(100):
    #             future = pool.submit(
    #                 schedule_pipe, [{"data": [data, data]}, {"data": [data, data]}])
    #         for i in range(100):
    #             inputB = {"data": [data, 2*data]}
    #             future = pool.submit(schedule_pipe, inputB)
    #             # futures.append(future)
    #             inputs_all.append(inputB)
    #         # final_result=[x.result() for x in futures]
    #     # 失败则此键值一定不存在，即使输入时已经存在。
    #     assert(torch.allclose(
    #         3*inputs_all[0]["result"][0], inputs_all[1]["result"][1]))


if __name__ == "__main__":
    pass

    a = TestBackend()

    a.setup_class()
    a.test_1()
    # a.test_infer()

    # pytest.main([__file__])
