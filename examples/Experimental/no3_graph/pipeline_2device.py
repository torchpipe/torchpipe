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


import cv2
import torch
import torchpipe as tp
from torchpipe import pipe, TASK_DATA_KEY, TASK_RESULT_KEY, TASK_BOX_KEY

tp.utils.cpp_extension.load(name="yolox", sources=["./yolox_new.cpp"])

parser = argparse.ArgumentParser()
parser.add_argument(
    "--config", dest="toml", type=str, default="./pipeline_2device.toml", help="configuration file"
)
parser.add_argument("--benchmark", action="store_true")
args = parser.parse_args()

if __name__ == "__main__":
    img_path = "../../../test/assets/norm_jpg/dog.jpg"
    img = cv2.imread(img_path, 1)
    img = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])[1]
    img = img.tobytes()

    toml_path = args.toml
    print(f"toml: {toml_path}")

 
    # 调用
    model = pipe(toml_path)

    def run(img_data, save_img=False):
        img_path, img_data = img_data[0]
        input = {TASK_DATA_KEY: img_data}
        input["node_name"] = "jpg_decoder"
        model(input)

        if save_img:
            print(input.keys())
            print(input["score_1"], input["score_2"], input["cls_1_result"],input["result"])

            detect_result = input[TASK_BOX_KEY]
            print(("detect_result: ", detect_result))

            img = cv2.imread(img_path)
            for t in range(len(detect_result)):
                x1, y1, x2, y2= detect_result[t].tolist()
                img = cv2.rectangle(
                    img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2
                )
            out_path = "dog_result.jpg"
            cv2.imwrite(out_path, img[:, :, ::1])
            print(f"result saved in: {out_path}")

    if args.benchmark:
        from torchpipe.utils import test

        test.test_from_raw_file(
            run,
            os.path.join("../../../test/assets/norm_jpg"),
            num_clients=40,
            total_number=20000,
        )
    else:
        run([(img_path, img)], save_img=True)
