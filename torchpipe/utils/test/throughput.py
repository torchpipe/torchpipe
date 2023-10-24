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
import numpy as np
import os
from .test_tools import test_function


class Model:
    def __init__(self, tp_model):
        self.tp = tp_model

    def __call__(self, data):
        input = {"data": data}

        self.tp(input)
        return input["result"]


def get_model(args_dict):
    import torchpipe as tp

    default_config = {
        "instance_num": "2",
        "max": "4",
        "batching_timeout": 5,
        "backend": "SyncTensor[TensorrtTensor]",
    }

    default_config.update(args_dict)
    args_dict.update(default_config)

    # if not 'model::cache' in args_dict and args_dict['model'].endswith(".onnx"):
    #     args_dict['model::cache'] = args_dict['model'].replace(".onnx",f"_{args_dict['precision']}.trt")

    try:
        model = tp.pipe(default_config)
    except Exception as e:
        print("Error : torchpipe initialize failed ")
        raise e

    return Model(model)


def get_trt_input_v1(model_path):
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.ERROR)
    trt.init_libnvinfer_plugins(logger, "")
    if os.path.isfile(model_path):
        with open(model_path, "rb") as infile:
            engine = trt.Runtime(logger).deserialize_cuda_engine(infile.read())
            if engine is None:
                print(f"Error : Failed loading {model_path}")
                return None

    input_shapes = []
    for idx in range(engine.num_bindings):
        is_input = engine.binding_is_input(idx)
        name = engine.get_binding_name(idx)
        op_type = engine.get_binding_dtype(idx)
        shape = engine.get_binding_shape(idx)
        print(f"is_input :{is_input}, name:{name}, op_type:{op_type}, shape:{shape}")
        if is_input and "profile" not in name:
            input_shapes.append(shape)

    if len(input_shapes) == 1:
        input_shape_list = input_shapes[0]
    else:
        input_shape_list = None

    input_shape_list[0] = 1

    return input_shape_list


def get_onnx_input(model_path):
    import torchpipe as tp

    shape = tp.infer_shape(model_path)

    shape[0][0] = 1
    return shape[0]


def get_trt_input(model_path):
    import torchpipe as tp

    shape = tp.infer_shape(model_path)

    shape[0][0] = 1

    return shape[0]


def get_onnx_input_v1(model_path):
    try:
        import onnx
    except ImportError:
        import sys
        from rich.text import Text

        command = [sys.executable, "-m", "pip", "install", "onnx"]
        print(
            Text(
                f"Installing onnx by `{' '.join(command)}`, please wait for a moment..",
                style="bold magenta",
            )
        )
        import subprocess

        subprocess.check_call(command)
        import onnx

    model = onnx.load(model_path)
    input_shape = model.graph.input[0].type.tensor_type.shape.dim
    input_shape_list = [dim.dim_value for dim in input_shape]
    input_shape_list[0] = 1

    return input_shape_list


def test_throughput(model_or_args_dict, num_clients=10, total_number=10000):
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

    if model_path.endswith(".trt"):
        input_shape = get_trt_input(model_path)
    elif model_path.endswith(".onnx"):
        input_shape = get_onnx_input(model_path)

    input_data = torch.randn(*input_shape).cuda()

    forward_function = lambda: model(input_data)
    result = test_function(
        forward_function, num_clients=num_clients, total_number=total_number
    )

    result.update(args_dict)
    return result


def test_throughput_from_timm(
    model_name, num_classes, num_clients, total_number, config_dict
):
    import timm

    assert "model" not in config_dict.keys()
    assert args.model in timm.list_models()

    if model_name.endswith(".onnx") or model_name.endswith(".trt"):
        model_path = model_name
    else:
        import tempfile, timm

        model_path = os.path.join(tempfile.gettempdir(), f"{model_name}.onnx")

        model_eval = timm.create_model(
            model_name, pretrained=False, num_classes=num_classes
        ).eval()

        # step 1: convert to onnx
        from ..models import onnx_export

        onnx_export(model_eval, model_path, torch.randn(1, 3, 224, 224))

    config_dict["model"] = model_path

    return test_throughput(
        config_dict,
        num_clients=num_clients,
        total_number=total_number,
    )


if __name__ == "__main__":
    #  python -m torchpipe.utils.test.throughput --model=resnet50  --config instance_num:2 max:4
    import argparse

    parser = argparse.ArgumentParser(description="Test throughout of TorchPipe")
    parser.add_argument(
        "--model",
        type=str,
        default="",
        help="model name from timm or path(.onnx, .trt)",
    )
    parser.add_argument("--num_classes", type=int, default=3, help="Number classes ")
    parser.add_argument("--num_clients", type=int, default=10, help="client_number")
    parser.add_argument(
        "--total_number", type=int, default=10000, help="total number of data"
    )
    parser.add_argument(
        "--config", nargs="*", type=lambda x: x.split(":"), dest="config"
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
        user_config,
    )
    print("\nresult:\n")
    print(result)
