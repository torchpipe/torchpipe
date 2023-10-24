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

import torchpipe
import torch
from torchvision import models
import os
import argparse


def test_throughput(
    pth_model, model_name, precision="fp16", num_clients=5, total_number=5000
):
    input = torch.randn(1, 3, 224, 224)
    tmpdir = tempfile.gettempdir()
    onnx_path = os.path.join(tmpdir, f"{model_name}.onnx")

    # step 1: convert to onnx
    torchpipe.utils.models.onnx_export(pth_model, onnx_path, input)

    # step 2: throughput test
    dict_args = {
        "model": onnx_path,
        "precision": precision,
        "instance_num": "4",
        "max": "1",
    }

    result = torchpipe.utils.test.throughput(
        dict_args,
        num_clients=num_clients,
        total_number=total_number,
    )
    return result


def get_args():
    parser = argparse.ArgumentParser(description="Test Speed of TorchPipe")
    parser.add_argument("--model_name", type=str, default="resnet50", help="model name")
    parser.add_argument("--num_classes", type=int, default=1000, help="Number classes ")
    parser.add_argument("--num_clients", type=int, default=10, help="client_number")
    parser.add_argument(
        "--total_number", type=int, default=10000, help="total number of data"
    )
    parser.add_argument("--precision", type=str, default="fp32", help="fp16, fp32")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    import timm
    import tempfile

    args = get_args()

    total_result = {}

    # test timm model
    timm_models_default = [
        "resnet18",
        "resnet50",
        "convnextv2_atto",
        "convnextv2_femto",
        "convnextv2_pico",
        "convnextv2_nano",
        "convnextv2_tiny",
    ]

    # fastervit model
    try:
        import fastervit

        fastervit_models = ["faster_vit_0_224", "faster_vit_1_224"]
    except ImportError:
        fastervit_models = []

    if args.model_name is not None:
        if args.model_name in timm.list_models():
            timm_models_default = [args.model_name]
            fastervit_models = []
        elif "faster_vit" in args.model_name:
            timm_models_default = []
            fastervit_models = [args.model_name]
        else:
            raise ValueError(f"{args.model_name} is not supported")

    for model_name in timm_models_default:
        m = timm.create_model(
            model_name, pretrained=False, num_classes=args.num_classes
        ).eval()
        result = test_throughput(
            m,
            model_name,
            precision=args.precision,
            num_clients=args.num_clients,
            total_number=args.total_number,
        )
        total_result[model_name] = result

    ## test fastervit

    for model_name in fastervit_models:
        m = fastervit.create_model(
            model_name, pretrained=False, num_classes=args.num_classes
        ).eval()
        result = test_throughput(
            m,
            model_name,
            precision=args.precision,
            num_clients=args.num_clients,
            total_number=args.total_number,
        )
        total_result[model_name] = result

    ### 将result以表格的形式写到markdown里面
    import pandas as pd

    df = pd.DataFrame(total_result)
    table = df.to_markdown(disable_numparse=True)
    print(table)
