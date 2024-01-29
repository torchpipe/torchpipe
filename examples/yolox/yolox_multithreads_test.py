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


#!coding=utf8
import torch
import cv2

import os
from torchpipe import pipe, TASK_DATA_KEY, TASK_RESULT_KEY, TASK_BOX_KEY
# import time
# time.sleep(6)
 

import numpy as np

import torchpipe as tp
from typing import List

curr_dir = os.path.dirname(__file__)
tp.utils.cpp_extension.load(name="yolox", sources=[os.path.join(curr_dir, "./yolox.cpp")])

class CustomSampler(tp.utils.test.RandomSampler):
    def __init__(self, model, data_source: List, batch_size=1):
        super().__init__(data_source, batch_size)
        self.model=model
        self.result = {}

    def forward(self, img_datas):
        inputs = []
        for img_path, img_data in img_datas:
            input = {TASK_DATA_KEY: img_data}
            input["node_name"] = "jpg_decoder"
            inputs.append(input)
        self.model(inputs)
        

        for i in range(len(inputs)):

            input = inputs[i]
            img_path, _ = img_datas[i]

            detect_result = input["result"]
 
            if img_path in self.result.keys():
                assert(len(detect_result) == len(self.result[img_path]))
                for j in range(len(detect_result)):
                    if not (torch.allclose(self.result[img_path][j], detect_result[j], atol=1e-3, rtol=1e-3)):
                        print(img_path, self.batchsize(), self.result[img_path][j], detect_result[j])
                        assert(False)
            else:
                self.result[img_path] = detect_result
            
def test_threads():
    img_path = os.path.join(curr_dir, "../../test/assets/norm_jpg/dog.jpg")
    img = cv2.imread(img_path, 1)
    img2 = cv2.resize(img, (224,224))
    img = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])[1].tobytes()
    img2 = cv2.imencode('.jpg', img2, [int(cv2.IMWRITE_JPEG_QUALITY), 95])[1].tobytes()

    toml_path = os.path.join(curr_dir, "./yolox.toml")
    print(f"toml: {toml_path}")

    # 调用
    model = pipe(toml_path)

    forwards = [CustomSampler(model, [(1,img),(2,img2)], i+1) for i in range(10)]

    tp.utils.test.test(forwards, 1000)
    
if __name__ == "__main__":
    test_threads()