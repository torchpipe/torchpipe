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
        import torchvision.models as models
        try:
            from torchvision.models.resnet import ResNet18_Weights

            self.resnet18 = (
                models.resnet18(
                    weights=ResNet18_Weights.IMAGENET1K_V1).eval().cuda()
            )
        except:
            self.resnet18 = models.resnet18(pretrained=True).eval().cuda()
        # resnet18 = models.resnet18(weights=None).eval().cuda()

        from torchpipe.tool import onnx_tools

        self.res_p = os.path.join(
            tempfile.gettempdir(), "./resnet18_with_mean_std.onnx"
        )

        onnx_tools.torch2onnx(
            self.resnet18,
            self.res_p,
            input_shape=(3, 224, 224),
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.120, 57.375],
        )

        self.resnet18.forward = self.resnet18._forward
        self.res_p_no_mean_std = os.path.join(
            tempfile.gettempdir(), "./resnet18_no_mean_std.onnx"
        )
        onnx_tools.torch2onnx(
            self.resnet18,
            self.res_p_no_mean_std,
            input_shape=(3, 224, 224),
            mean=None,
            std=None,
        )

        jpg_path = "assets/encode_jpeg/grace_hopper_517x606.jpg"

        self.img = cv2.imread(jpg_path)

        assert self.img is not None

        # img = self.img
        self.resize_224_img = cv2.resize(self.img, (224, 224))

        onnxruntime = pytest.importorskip("onnxruntime")
        ort_session = onnxruntime.InferenceSession(
            self.res_p, providers=["CPUExecutionProvider"]
        )
        # ort_session.set_providers(['CPUExecutionProvider'])

        img_onnx = cv2.cvtColor(self.resize_224_img, cv2.COLOR_BGR2RGB)
        img_onnx = cv2.resize(img_onnx, (224, 224)).transpose(2, 0, 1)
        ort_inputs = {
            ort_session.get_inputs()[0].name: np.expand_dims(
                img_onnx.astype(np.float32), 0
            )
        }
        ort_outs = ort_session.run(None, ort_inputs)
        classification = torch.from_numpy(ort_outs[0]).cuda()
        classification = torch.softmax(classification, 1)
        self.max_onnx = (torch.max(classification[0])).item()

    def decode_run(self, input_dict, target, model=None):
        if model:
            model(input_dict)
        else:
            assert False
        assert input_dict["color"] in [b"rgb", b"bgr"]

        if input_dict["color"] == b"rgb":
            input_dict[TASK_RESULT_KEY] = input_dict[TASK_RESULT_KEY][
                :, [2, 1, 0], :, :
            ]
        assert len(input_dict[TASK_RESULT_KEY].squeeze(0)) == 3
        input_dict[TASK_RESULT_KEY] = (
            input_dict[TASK_RESULT_KEY].squeeze(0).permute(1, 2, 0)
        )
        z = target - input_dict[TASK_RESULT_KEY].float()
        rel = torch.mean(z).item()

        assert abs(rel) < 1

    @pytest.mark.parametrize("device", ["cpu", "gpu"])
    def test_nvjpeg_or_opencv(self, device):
        config = {"jpg_decoder": {"device": device}}
        if device == "cpu":
            config = {
                "jpg_decoder": {
                    "device": device,
                    "backend": "Sequential[DecodeTensor,Tensor2Mat,Mat2Tensor,SyncTensor]",
                }
            }
        else:
            config = {
                "jpg_decoder": {
                    "device": device,
                    "backend": "Sequential[DecodeMat,Mat2Tensor,SyncTensor]",
                }
            }

        model = pipe(config)

        jpg_path = "assets/encode_jpeg/grace_hopper_517x606.jpg"
        img = cv2.imread(jpg_path)
        target = torch.from_numpy(img).float()
        if device == "gpu":
            target = target.cuda()

        with open(jpg_path, "rb") as f:
            raw_jpg = f.read()

        data = np.frombuffer(raw_jpg, dtype=np.uint8).copy()

        input_dict = {TASK_DATA_KEY: data.tobytes()}

        self.decode_run(input_dict, target, model)

        input_dict = {TASK_DATA_KEY: data.tobytes(
        ), TASK_RESULT_KEY: "56789876567"}
        self.decode_run(input_dict, target, model)

        with pytest.raises(RuntimeError):
            input_dict = {TASK_DATA_KEY: torch.from_numpy(data)}
            model(input_dict)

        input_dict = {
            TASK_DATA_KEY: "0005678989876789---08768765678000",
            TASK_RESULT_KEY: "-",
        }
        model(input_dict)
        assert TASK_RESULT_KEY not in input_dict.keys()

        with pytest.raises(RuntimeError):
            input_dict = {TASK_DATA_KEY: 1}
            model(input_dict)
        # assert TASK_RESULT_KEY not in input_dict.keys()

        with pytest.raises(IndexError):
            input_dict = {}
            model(input_dict)

        img_dir = "assets/damaged_jpeg/"
        for i in os.listdir(img_dir):
            jpg_path = os.path.join(img_dir, i)

            with open(jpg_path, "rb") as f:
                raw_jpg = f.read()

            input_dict = {TASK_DATA_KEY: raw_jpg}
            model(input_dict)
            if TASK_RESULT_KEY in input_dict.keys():
                assert (
                    len(input_dict[TASK_RESULT_KEY].shape) == 4
                    and input_dict[TASK_RESULT_KEY].shape[1] != 0
                )
            else:
                pass

    def test_tensorrt_resnet18(self):
        max_onnx = self.max_onnx
        img = self.resize_224_img

        # config = {'resnet18':
        #   {'model': self.res_p, "resize_type": "resize"}}  # default color : rgb

        config = {
            "resnet18": {
                "backend": "Sequential[Tensor2Mat,cvtColorMat,Mat2Tensor,TensorrtTensor, SyncTensor]",
                "model": self.res_p,
                "color": "rgb",
            }
        }

        models = pipe(config)

        target = torch.from_numpy(img)

        max_result = []
        input_dict = {TASK_DATA_KEY: target, "color": "bgr"}
        models(input_dict)
        assert TASK_RESULT_KEY in input_dict.keys()
        input_dict[TASK_RESULT_KEY] = torch.softmax(
            input_dict[TASK_RESULT_KEY], 1)
        max_result.append((torch.max(input_dict[TASK_RESULT_KEY])).item())

        input_dict = {TASK_DATA_KEY: target.cuda(), "color": "bgr"}
        models(input_dict)
        input_dict[TASK_RESULT_KEY] = torch.softmax(
            input_dict[TASK_RESULT_KEY], 1)
        max_a = (torch.max(input_dict[TASK_RESULT_KEY])).item()
        max_result.append(max_a)

        input_dict = {TASK_DATA_KEY: target[:, :, [2, 1, 0]], "color": "rgb"}
        models(input_dict)
        input_dict[TASK_RESULT_KEY] = torch.softmax(
            input_dict[TASK_RESULT_KEY], 1)
        max_a = (torch.max(input_dict[TASK_RESULT_KEY])).item()
        max_result.append(max_a)

        input_dict = {
            TASK_DATA_KEY: torch.unsqueeze(target[:, :, [2, 1, 0]], 0),
            "color": "rgb",
        }
        with pytest.raises(RuntimeError):
            models(input_dict)
        # input_dict[TASK_RESULT_KEY] = torch.softmax(
        #     input_dict[TASK_RESULT_KEY], 0)
        # max_a = (torch.max(input_dict[TASK_RESULT_KEY])).item()
        # max_result.append(max_a)

        for i in max_result:
            assert abs(i - max_onnx) < 0.01, f"{i}, {len(max_result)}"

    def test_tensorrt_renet18_mean_std(self):
        config = {
            "resnet18": {
                "backend": "Sequential[Tensor2Mat,cvtColorMat,Mat2Tensor,TensorrtTensor, SyncTensor]",
                "model": self.res_p_no_mean_std,
                "color": "rgb",
                "mean": "123.675, 116.28, 103.53",
                "std": "58.395, 57.120, 57.375",
            }
        }

        target = torch.from_numpy(self.resize_224_img)

        max_result = []
        input_dict = {TASK_DATA_KEY: target, "color": "bgr"}
        models = pipe(config)
        models(input_dict)
        input_dict[TASK_RESULT_KEY] = torch.softmax(
            input_dict[TASK_RESULT_KEY], 1)
        max_a = (torch.max(input_dict[TASK_RESULT_KEY])).item()
        print(max_a, self.max_onnx)
        assert abs(max_a - self.max_onnx) < 0.01

    def test_tensorrt_resize(self):
        config = {
            "resnet18": {
                "backend": "Sequential[Tensor2Mat,ResizeMat,cvtColorMat,Mat2Tensor,TensorrtTensor, SyncTensor]",
                "model": self.res_p_no_mean_std,
                "resize_h": "224",
                "resize_w": "224",
                "color": "rgb",
                "precision": "fp32",
                # "save_dir":"./",
                "mean": "123.675, 116.28 , 103.53",
                "std": "58.395, 57.12, 57.375",
            }
        }

        img = self.img
        target = torch.from_numpy(img)

        max_result = []
        input_dict = {
            TASK_DATA_KEY: target,
            "color": "bgr",
            "node_name": "node_name_none",
        }
        models = pipe(config)
        models(input_dict)

        input_dict[TASK_RESULT_KEY] = torch.softmax(
            input_dict[TASK_RESULT_KEY], 1)
        # print(input_dict[TASK_RESULT_KEY])
        max_a = (torch.max(input_dict[TASK_RESULT_KEY])).item()
        print(max_a, self.max_onnx)
        assert abs(max_a - self.max_onnx) < 0.01

    def test_version(self):
        z = torchpipe.__version__
        assert z.startswith("0.")
        print(z)


if __name__ == "__main__":
    import time

    # time.sleep(10)
    a = TestBackend()
    a.setup_class()
    # a.test_nvjpeg_or_opencv("cpu")
    a.test_tensorrt_renet18_mean_std()
    # a.test_version()
    # a.test_nvjpeg_or_opencv("cpu")
    # a.test_nvjpeg_or_opencv("gpu")
    # a.test_tensorrt_renet18()
    # a.test_tensorrt_resize()
    # pytest.main([__file__])
