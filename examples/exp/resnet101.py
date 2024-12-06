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


import torch
import cv2
import os
import torchpipe as tp


import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
    "--config",
    dest="config",
    type=str,
    default="./cfg/resnet101.toml",
    help="configuration file",
)
parser.add_argument(
    "--client", dest="client", type=int, default=40, help="number of clients"
)

args = parser.parse_args()


def export_onnx(onnx_save_path):
    import torch
    import torchvision.models as models

    resnet101 = models.resnet101().eval()
    x = torch.randn(1, 3, 224, 224)
    onnx_save_path = "./resnet101.onnx"
    tp.utils.models.onnx_export(resnet101, onnx_save_path, x)
    # torch.onnx.export(resnet101,
    #                 x,
    #                 onnx_save_path,
    #                 opset_version=17,
    #                 do_constant_folding=True,
    #                 input_names=["input"],            # 输入名
    #                 output_names=["output"],  # 输出名
    #                 dynamic_axes={"input":{0:"batch_size"},"output":{0:"batch_size"}})


if __name__ == "__main__":

    import time

    # time.sleep(5)
    # prepare data:

    onnx_save_path = "./resnet101.onnx"
    if not os.path.exists(onnx_save_path):
        export_onnx(onnx_save_path)

    img_path = "../../test/assets/image/gray.jpg"
    img = open(img_path, "rb").read()

    toml_path = args.config

    from torchpipe import pipe, TASK_DATA_KEY, TASK_RESULT_KEY

    nodes = pipe(toml_path)

    def run(img):
        for img_path, img_bytes in img:
            input = {TASK_DATA_KEY: img_bytes, "node_name": "jpg_decoder"}
            nodes(input)

            if TASK_RESULT_KEY not in input.keys():
                print("error : no result")
                return
            z = input[TASK_RESULT_KEY].cpu()

    run([(img_path, img)])

    from torchpipe.utils.test import test_from_raw_file

    total_number = 20000
    if args.client == 1:
        total_number = 1000
    result = test_from_raw_file(
        run,
        os.path.join("../..", "test/assets/encode_jpeg/"),
        num_clients=args.client,
        request_batch=1,
        total_number=total_number,
    )

    # keep = ["throughput::qps", "latency::TP99", "latency::TP50", "cpu_usage", 'gpu_usage']
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


