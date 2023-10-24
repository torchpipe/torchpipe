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
import cv2
import os
import torchpipe as tp


import argparse

parser = argparse.ArgumentParser()


args = parser.parse_args()

if __name__ == "__main__":
    import time
    time.sleep(5)
    # prepare data:
    img_path = "../../test/assets/image/gray.jpg"
    img = open(img_path, 'rb').read()
    input_img = cv2.resize(cv2.imread(img_path), (224, 224))

    from torchpipe import pipe, TASK_DATA_KEY, TASK_RESULT_KEY
    nodes = pipe({"backend": "TensorSync[TensorrtTensor]",
                  "instance_num": 2,
                  "max": 4,
                  "batching_timeoyt": 5,
                  "model": "../../test/assets/resnet18.onnx",
                  "model::cache": "resnet18.trt"})

    input_tensor = torch.from_numpy(input_img).cuda()

    def only_model_run(img):
        input = {TASK_DATA_KEY: input_tensor, "node_name": "jpg_decoder"}
        nodes(input)

        if TASK_RESULT_KEY not in input.keys():
            print("error : no result")
            return
        return input[TASK_RESULT_KEY]

    from torchpipe.utils import test
    test.test_from_raw_file(only_model_run, os.path.join(
        "../..", "test/assets/encode_jpeg/"))
