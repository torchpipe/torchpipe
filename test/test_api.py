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
from multiprocessing.sharedctypes import Value
from torch import nn
import pytest
from torchpipe import TASK_BOX_KEY
from typing import Union, Tuple, List, Any
import torch
from torchpipe import pipe

# batch维度动态化的 tensorrt 模型推理

import time


# time.sleep(10)
def torch_avg_diff(torch_a, torch_b):
    a = torch.mean(torch.abs(torch_a - torch_b)).item()
    return a


class tensorrttensor:
    def __init__(
        self,
        model: Union[bytes, str],
        precision: str = "fp16",
        batch_size: Union[str, int] = 1,
        instance_num: Union[str, int] = 1,
        batching_timeout: Union[str, int] = 0,
        color: str = "rgb",
        mean="",  # "123.675, 116.28, 103.53"
        std="",
        save_dir="",
    ):  # "58.395, 57.120, 57.375"
        self.pipe = pipe(
            {
                "resnet": {
                    "model": model,  # onnx模型路径，以.onnx结尾
                    "precision": precision,  # 机器如果不支持，回跳到fp32
                    "instance_num": instance_num,  # 实例数目，Union[int, str]
                    "batching_timeout": batching_timeout,  # 超时时间
                    "color": color,  # 模型接收的颜色通道顺序
                    "batch_size": batch_size,
                    "mean": mean,
                    "std": std,
                    "backend": "SyncTensor[TensorrtTensor]",
                    "save_dir": save_dir,
                }
            }
        )

    def __call__(self, data: torch.Tensor, color: str = "rgb"):
        context = {"data": data, "color": color}
        self.pipe(context)
        # List[torch.Tensor]; 自定义后处理可能带来新的输出类型和自定义键值， 比如yolox
        return context["result"]

    # 注意： 除了类似检测模型输出多个新目标这类新增上下文的场景，不推荐使用multi_infer接口；
    # multi_infer一次送入多个数据可能会造成不必要地互相等待。
    # 我们建议在一个串行流程（如单张图片解码+模型A+模型B+模型C）中，一个数据流跑完整个流程；
    # 封装整个串行过程为一个函数，对此函数进行异步或者线程池的并行推理， 以便达到节点级的微服务化。
    # 对于新增上下文的场景， 如果新的上下文后面还有2个或以上串行节点，调用multi_infer也将在
    # 数据进入后面第二个节点时等待所有数据跑完第一个节点。消解此等待可使用流水线并行设施
    def multi_infer(
        self, data: Union[torch.Tensor, List[torch.Tensor]], color: str = "rgb"
    ) -> dict:
        context = []
        for single_data in data:  # 分拆batch维度
            context.append({"data": single_data, "color": color})
        self.pipe(context)
        # or self.pipe(context, color=color)
        return context


# jpg解码


class nvjpeg_or_opencv:
    def __init__(self, device: str = "gpu", out_device: str = "gpu"):
        self.pipe = pipe(
            {
                "device": device,
                "out_device": out_device,
                "backend": "SyncTensor[DecodeTensor]",
            }
        )

    def __call__(self, data: Union[str, bytes]) -> Tuple[torch.Tensor, str]:
        input = {"data": data}
        self.pipe(input)
        return input["result"], input["color"]

    def multi_infer(self, data: List[Union[str, bytes]]) -> List[dict]:
        input = []
        for single_data in data:
            input.append({"data": single_data})
        self.pipe(input)
        # 注意， 可能个别输入解码失败， 导致无result键值; 在color上输出rgb或者bgr
        # return [x["result"] for x in input]
        return input  # 解码不一定全部成功。失败时，对应dict无 "result" 键值


# 抠图
class CropTensor:
    def __init__(self, instance_num: Union[str, int] = 1):
        self.pipe = pipe(
            {"backend": "CropTensor", "instance_num": instance_num})

    def __call__(
        self, data: torch.Tensor, box: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        input = {"data": data, TASK_BOX_KEY: box}
        self.pipe(input)
        return input["result"]


class Identity(torch.nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        self.identity = torch.nn.Identity()

    def forward(self, x):
        x = self.identity(x)
        return x


class TestBackend:
    @classmethod
    def setup_class(self):
        self.identity_model = Identity().eval()

        self.data_bchw = torch.rand((1, 3, 224, 224))
        self.data_chw = self.data_bchw.squeeze(0)

        self.data_2bchw = torch.rand((2, 3, 224, 224))

        self.data_wc = torch.rand((224, 3))

        self.data_bhwc = torch.rand((1, 224, 224, 3))
        self.data_hwc = self.data_bhwc.squeeze(0)

        self.shape_bhwc = torch.rand((1, 111, 122, 3)) * 100
        self.shape_hwc = self.data_bhwc.squeeze(0)

        out_file = "./assets/temp.onnx"
        torch.onnx.export(
            self.identity_model,
            self.data_bchw,
            out_file,
            opset_version=13,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={
                "input": {0: "batch_size"},  # 批处理变量
                "output": {0: "batch_size"},
            },
        )

        self.tensorrt = tensorrttensor(out_file)
        # try:
        #     re = self.tensorrt(self.data_dims3 )
        # except Exception as e:
        #     print(e, type(e))

        # print(torch.equal(self.data_dims3, re[0].cpu()))
        # print(re)

    def test_infer(self):
        # bchw input
        result_4_trt = self.tensorrt(self.data_bchw, color="bgr")
        assert isinstance(result_4_trt, torch.Tensor)
        # batch=1, remove batch dim
        assert torch.equal(result_4_trt.cpu()[0], self.data_bchw[0])

        # bhwc input
        result_4_trt = self.tensorrt(self.data_bhwc, color="bgr")
        assert isinstance(result_4_trt, torch.Tensor)

        b = self.data_bhwc.permute((0, 3, 1, 2))
        print(b.shape, b[0, 0, 222:, 222:])
        print(result_4_trt.cpu()[0, 0, 222:, 222:], " :::: ")
        assert torch.equal(
            result_4_trt.cpu(), self.data_bhwc.permute((0, 3, 1, 2))
        )  # batch=1, remove batch dim

        # hwc input
        result_4_trt = self.tensorrt(self.data_hwc, color="bgr")
        assert isinstance(result_4_trt, torch.Tensor)
        assert torch.equal(result_4_trt.cpu()[
                           0], self.data_hwc.permute((2, 0, 1)))

        # chw input
        result_4_trt = self.tensorrt(self.data_chw, color="bgr")
        assert isinstance(result_4_trt, torch.Tensor)
        assert torch.equal(result_4_trt.cpu()[0], self.data_chw)

        # wc input
        with pytest.raises(RuntimeError):
            result_4_trt = self.tensorrt(self.data_wc, color="bgr")

    def test_init_error(self):
        onnx_bytes = "not_exists_file.onnx"
        with pytest.raises(RuntimeError):
            tensorrttensor(onnx_bytes)

        file_name = "not_exists_file.trt"
        with pytest.raises(RuntimeError):
            tensorrttensor(file_name)

        file_name = "not_exists_file.zz"
        with pytest.raises(RuntimeError):
            tensorrttensor(file_name)

        file_name = b"not_exists_file.zz"
        with pytest.raises(RuntimeError):
            tensorrttensor(file_name)

    def test_infer_error(self):
        with pytest.raises(RuntimeError):
            self.tensorrt(self.shape_bhwc)

        with pytest.raises(RuntimeError):
            self.tensorrt(self.data_wc)

    def test_infer_float_uint8(self):
        float_result = self.tensorrt(self.data_bhwc)[0]
        uint8_result = self.tensorrt(self.data_bhwc.to(torch.uint8))[0]

        # assert(torch.equal(float_result, uint8_result))

        a = torch_avg_diff(float_result, uint8_result)
        assert a < 1


if __name__ == "__main__":
    import time

    time.sleep(5)

    a = TestBackend()

    a.setup_class()

    # print(a, "--")

    a.test_init_error()
    # a.test_infer_error()
    # a.test_infer_float_uint8()
    a.test_infer()
    # pytest.main([__file__])
