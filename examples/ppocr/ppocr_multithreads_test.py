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
import torch
import cv2
from PIL import Image
import numpy as np
from draw_ocr import draw_ocr_box_txt
import os
from torchpipe import pipe, TASK_DATA_KEY, TASK_RESULT_KEY, TASK_BOX_KEY
# import time
# time.sleep(6)
 

import numpy as np

import torchpipe as tp
from typing import List


import torchpipe as tp
import glob, logging
cwd = os.path.dirname(os.path.abspath(__file__))
files = glob.glob(os.path.join(cwd, "csrc/src/postprocess_op.cpp"))
files += glob.glob(os.path.join(cwd, "csrc/ocr.cpp"))
files += glob.glob(os.path.join(cwd, "csrc/src/utility.cpp"))


print("compile: ", files)
tp.utils.cpp_extension.load(name="dbnet", sources=files,
                  extra_include_paths=[os.path.join(cwd, "csrc/"),
                                       os.path.join(cwd, "ppocr/cpp/"), "/usr/local/include/opencv4/"],
                  extra_ldflags=["-lopencv_core", "-lopencv_imgcodecs", "-lopencv_imgproc"], rebuild_if_exist=True, is_python_module=False)


logging.basicConfig(level=logging.DEBUG)


class CustomSampler(tp.utils.test.RandomSampler):
    def __init__(self, model, data_source: List, batch_size=1):
        super().__init__(data_source, batch_size)
        self.model=model
        self.result = {}

    def forward(self, img_datas):
        inputs = []
        for img_path, img_data in img_datas:
            input = {}
            input[TASK_DATA_KEY] = img_data
            input["color"] = "bgr"
            input["node_name"] = "jpg_decoder"

            inputs.append(input)
        self.model(inputs)
        

        for i in range(len(inputs)):

            input = inputs[i]
            img_path, _ = img_datas[i]

            boxes = input[TASK_BOX_KEY]
            scores = input["scores"]
            texts = []
            for text in input[TASK_RESULT_KEY]:
                texts.append(text.decode("utf8"))
                
            
            detect_result = [boxes, np.asanyarray(scores), texts]

            if img_path in self.result.keys():
                for i in range(len(detect_result[0])):
                    for j in range(len(detect_result[0][i])):
                        assert np.array_equal(detect_result[0][i][j], self.result[img_path][0][i][j])
                if img_path == 1:
                    for t in detect_result:
                        assert(len(t) == 37) # 37
                elif img_path == 2:
                    for t in detect_result:
                        assert(len(t) == 23)  # 23    
                else:
                    print("error, img_path= ", img_path)   
                    print([len(x) for x in detect_result], [len(x) for x in self.result[img_path]])  
                    assert(False)   
                    exit(0)  
                
                # is not supposed to be the same, as batchindex and width changed     
                if (not np.allclose(self.result[img_path][1][:-5], detect_result[1][:-5],atol=1e-1,rtol=1e-1)):
                    t = [] # VisualizEd:  VISUALIZED : detect_result[2][-3]
                    for zz in range(len(self.result[img_path][1])):
                        t.append(self.result[img_path][1][zz] - detect_result[1][zz])
                    print("score not the same, t = ", t)
                    print("\n\n")
                    print("--"*10)
                    print(self.result[img_path][0]," | \n",  detect_result[0], "\n")
                    print(self.result[img_path][1]," | \n",  detect_result[1])
                    print(self.result[img_path][2]," | \n",  detect_result[2])
                    # assert(False)
                diff_size = 0
                for text in range(len(self.result[img_path][2])-4):
                    if self.result[img_path][2][text] != detect_result[2][text]:
                        diff_size += 1
                if diff_size*1.0/(len(self.result[img_path][2])-4)> 0.7:
                    print("error, diff_size", diff_size)
                    print(self.result[img_path][2], detect_result[2])
                    assert(False)
                    # exit(0)

            else:
                self.result[img_path] = detect_result
            
# def test_threads():
#     img_path = os.path.join(curr_dir, "../../test/assets/norm_jpg/dog.jpg")
#     img = cv2.imread(img_path, 1)
#     img2 = cv2.resize(img, (224,224))
#     img = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])[1].tobytes()
#     img2 = cv2.imencode('.jpg', img2, [int(cv2.IMWRITE_JPEG_QUALITY), 95])[1].tobytes()

#     toml_path = os.path.join(curr_dir, "./yolox.toml")
#     print(f"toml: {toml_path}")

#     # 调用
#     model = pipe(toml_path)

#     forwards = [CustomSampler(model, [(1,img),(2,img2)], i+1) for i in range(10)]

#     tp.utils.test.test(forwards, 1000)



def test_threads():

    # prepare data:
    img_path = "../../test/assets/image/lite_demo.png"
    # img_path = "./ppocr/lite_demo.png"
    gimg = cv2.imread(img_path, 1)

    img2 = cv2.resize(gimg, (224,224))

    import time
    # time.sleep(5)

    # 调用
    from torchpipe import pipe, TASK_DATA_KEY, TASK_RESULT_KEY, TASK_BOX_KEY
    model = pipe("ocr.toml")
    # for i in range(1*1):
    #     input = {}

    #     input[TASK_DATA_KEY] = torch.from_numpy(gimg).permute(2,0,1).unsqueeze(0)
    #     input["color"] = "bgr"
    #     input["node_name"] = "jpg_decoder"

    #     model(input)
    gimg = torch.from_numpy(gimg).permute(2,0,1).unsqueeze(0).cuda()
    img2 = torch.from_numpy(img2).permute(2,0,1).unsqueeze(0).cuda()
    forwards = [CustomSampler(model, [(1,gimg),(2,img2)], i+1) for i in range(10)]    
    tp.utils.test.test(forwards, 100)
 
    


if __name__ == "__main__":
    test_threads()