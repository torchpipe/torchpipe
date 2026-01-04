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

#!coding=utf8
import os
import time
import argparse
from types import MethodType


import cv2
import torch
import torchpipe as tp

tp.utils.cpp_extension.load(name="yolox", sources=["./yolox_new.cpp"])

from torchpipe import pipe, TASK_DATA_KEY, TASK_RESULT_KEY, TASK_BOX_KEY


parser = argparse.ArgumentParser()
parser.add_argument(
    "--config",
    dest="toml",
    type=str,
    default="./pipeline_gpu.toml",
    help="configuration file",
)
parser.add_argument(
    "--client", dest="client", type=int, default=20, help="number of clients"
)
parser.add_argument(
    "--total_number",
    dest="total_number",
    type=int,
    default=10000,
    help="number of clients",
)
parser.add_argument(
    "--preprocess",
    dest="preprocess",
    type=str,
    default="cpu",
    help="cpu preprocess or gpu preprocess",
)
parser.add_argument(
    "--preprocess-instances",
    dest="preprocess_instances",
    type=int,
    default=0,
    help="cpu preprocess or gpu preprocess instance",
)
parser.add_argument("--benchmark", action="store_true")
args = parser.parse_args()


def wrap_argmax(model):
    def forward(self, input):
        output = self._forward(input)
        output = self._forward(input)
        return torch.max(output.softmax(-1), 1)

    model._forward = model.forward
    model.forward = MethodType(forward, model)


def export_timm_onnx(model_name):
    onnx_save_path = f"./{model_name}.onnx"
    if os.path.exists(onnx_save_path):
        return
    import torch
    import torchvision.models as models
    import timm

    # resnet101 = tp.utils.models.create_model(name).eval()
    if model_name in timm.list_models():
        model = timm.create_model(model_name, pretrained=True, exportable=True).eval()
    else:
        if "faster_vit" in model_name:
            import fastervit

            assert timm.__version__ == "0.9.6"
            # timm==0.9.6

            model = fastervit.create_model(model_name, pretrained=False).eval()
    x = torch.randn(1, 3, 224, 224)
    wrap_argmax(model)
    tp.utils.models.onnx_export(model, onnx_save_path, x, opset=17)


def export_fastervit_onnx():

    onnx_name = os.path.join(".", "fastervit_0_224_224.onnx")
    if os.path.exists(onnx_name):
        return
    import fastervit, onnx

    model = fastervit.create_model(
        "faster_vit_0_224", resolution=224, pretrained=True, exportable=True
    )

    wrap_argmax(model)
    tp.utils.models.onnx_export(model, onnx_name, None, 17)
    import onnxsim

    for i in range(3):
        model_simp, check = onnxsim.simplify(onnx_name)
        assert check, "Simplified ONNX model could not be validated"
        onnx.save(model_simp, onnx_name)


if __name__ == "__main__":

    img_path = "../../../test/assets/norm_jpg/dog.jpg"
    img = cv2.imread(img_path, 1)
    img = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])[1]
    img = img.tobytes()

    toml_path = args.toml
    print(f"toml: {toml_path}")

    export_fastervit_onnx()
    export_timm_onnx("resnet50")
    export_timm_onnx("resnet18")
    # 调用
    model = pipe(toml_path)

    def run(img_data, save_img=False):
        img_path, img_data = img_data[0]
        input = {TASK_DATA_KEY: img_data}
        input["node_name"] = "jpg_decoder"
        model(input)

        if save_img:
            print(input.keys())
            print(
                input["score_r"],
                input["score_vit"],
                input["r18_result"],
                input["result"],
            )

            detect_result = input[TASK_BOX_KEY]
            print(("detect_result: ", detect_result))

            img = cv2.imread(img_path)
            for t in range(len(detect_result)):
                x1, y1, x2, y2 = detect_result[t].tolist()
                img = cv2.rectangle(
                    img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2
                )
            out_path = "dog_result.jpg"
            cv2.imwrite(out_path, img[:, :, ::1])
            print(f"result saved in: {out_path}")

    if args.benchmark:
        from torchpipe.utils import test

        result = test.test_from_raw_file(
            run,
            os.path.join("../../../test/assets/norm_jpg"),
            num_clients=args.client,
            total_number=args.total_number,
        )
        keep = {
            "throughput::qps": "QPS",
            "latency::TP99": "TP99",
            "latency::TP50": "TP50",
            "gpu_usage": "GPU Usage",
        }

        print("\n", result)

        new_result = {keep[k]: v for k, v in result.items() if k in keep.keys()}
        print("\n\n")
        print({args.client: new_result})
        print("\n\n")

    else:
        run([(img_path, img)], save_img=True)
