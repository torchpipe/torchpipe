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

# import torch
import numpy as np
import os
from .test import test_from_ids


class Model:
    def __init__(self, tp_model):
        self.tp = tp_model

    def __call__(self, data):
        input = {"data": data}

        self.tp(input)
        return input["result"]


def get_model(args_dict):
    import torchpipe as tp
    import omniback

    default_config = {
        "instance_num": "2",
        "max": "4",
        "batching_timeout": 5,
        "backend": "StreamGuard[TensorrtTensor]",
    }

    default_config.update(args_dict)
    args_dict.update(default_config)

    try:
        print(default_config)
        model = omniback.pipe(default_config)
    except Exception as e:
        print("Error : torchpipe initialize failed ")
        raise e

    return Model(model)


def test_throughput(model_or_args_dict, num_clients=10, total_number=10000, input_shape=(224, 224)):
    if isinstance(model_or_args_dict, str):
        args_dict = {"model": model_or_args_dict}
    else:
        args_dict = model_or_args_dict
    if "model" not in args_dict:
        raise ValueError("args_dict must have [ model ]")

    model_path = args_dict["model"]

    if not (model_path.endswith(".onnx") or model_path.endswith(".trt")):
        raise ValueError("model_path must end with [.onnx] or [.trt] ")

    model = get_model(args_dict)

    input_shape = [1, 3] + list(input_shape)
    import torch

    input_data = torch.randn(*input_shape).cuda()

    def forward_function(ids): return model(input_data)
    result = test_from_ids(
        [forward_function]*num_clients, ids=[0]*total_number
    )

    result.update(args_dict)
    return result


def parse_config_arg(arg):
    if ":" in arg:
        parts = arg.rsplit(":", 1)  # 从右边开始分割，只分割一次
        if len(parts) == 2:
            key, value = parts
            return key, value
    raise ValueError(
        f"Invalid config format: '{arg}'. Expected 'key:value' format.")


def test_throughput_from_timm(
    model_name, num_classes, num_clients, total_number, input_shape, config_dict
):
    assert "model" not in config_dict.keys()

    if model_name.endswith(".onnx") or model_name.endswith(".trt"):
        model_path = model_name
    else:
        assert args.model in timm.list_models()
        import tempfile
        import timm

        model_path = os.path.join(tempfile.gettempdir(), f"{model_name}.onnx")

        model_eval = timm.create_model(
            model_name, pretrained=False, num_classes=num_classes
        ).eval()

        # step 1: convert to onnx
        torchpipe.utils.models.onnx_export(
            resnet101, model_path, torch.randn(1, 3, 224, 224))

    config_dict["model"] = model_path

    return test_throughput(
        config_dict,
        num_clients=num_clients,
        total_number=total_number,
        input_shape=input_shape,
    )


if __name__ == "__main__":
    #  python -m torchpipe.utils.throughput --model=resnet50  --config instance_num:2 max:4
    import argparse
    # import time
    # time.sleep(5)

    import omniback
    omniback.init("DebugLogger")

    parser = argparse.ArgumentParser(
        description="Test throughout of TorchPipe")
    parser.add_argument(
        "--model",
        type=str,
        default="",
        help="model name from timm or path(.onnx, .trt)",
    )
    parser.add_argument("--num_classes", type=int,
                        default=3, help="Number classes ")
    parser.add_argument("--num_clients", type=int,
                        default=10, help="client_number")
    parser.add_argument(
        "--total_number", type=int, default=10000, help="total number of data"
    )
    parser.add_argument(
        "--config", nargs="*", type=parse_config_arg, dest="config"
    )
    # add h w (must supply)
    parser.add_argument(
        "--input_shape",
        type=int,
        nargs=2,
        default=(224, 224),
        help="input shape (height, width)",
    )

    args = parser.parse_args()

    user_config = dict(args.config)

    user_config.update(
        {
            # put your config here
        }
    )

    result = test_throughput_from_timm(
        args.model,
        args.num_classes,
        args.num_clients,
        args.total_number,
        args.input_shape,
        user_config,
    )
    print("\nresult:\n")
    print(result)
