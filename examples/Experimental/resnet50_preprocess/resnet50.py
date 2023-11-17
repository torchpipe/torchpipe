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

parser.add_argument('--config', dest='config', type=str,  default="./resnet50.toml",
                    help='configuration file')

args = parser.parse_args()


def export_onnx(onnx_save_path):
    import torch
    import torchvision.models as models
    resnet50 = models.resnet50().eval()
    x = torch.randn(1,3,224,224)
    onnx_save_path = "./resnet50.onnx"
    tp.utils.models.onnx_export(resnet50, onnx_save_path, x)
    # torch.onnx.export(resnet50,
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


    onnx_save_path = "./resnet50.onnx"
    if not os.path.exists(onnx_save_path):
        export_onnx(onnx_save_path)

    img_path = "../../../test/assets/image/gray.jpg"
    img=open(img_path,'rb').read()

    toml_path = args.config 
    
    from torchpipe import pipe, TASK_DATA_KEY, TASK_RESULT_KEY
    nodes = pipe(toml_path)

    def run(img):
        img_path, img_bytes = img[0]
 
        input = {TASK_DATA_KEY: img_bytes, "node_name": "jpg_decoder"}
         
        nodes(input)

        input[TASK_RESULT_KEY].argmax().item()
        max_score, max_class = torch.max(input[TASK_RESULT_KEY], dim=1)
        max_score_float = max_score.item()
        max_class_int = max_class.item()
        return max_class_int, max_score_float


    run([(img_path, img)])

    from torchpipe.utils.test import test_from_raw_file
    result = test_from_raw_file(run, os.path.join("../../..", "test/assets/encode_jpeg/"),num_clients=40, batch_size=1,total_number=20000)


    print("\n", result)

    import pickle
    pkl_path = toml_path.replace(".toml",".pkl")
    with open(pkl_path,"wb") as f:
        pickle.dump(result, f)
    print("save result to ", pkl_path)
