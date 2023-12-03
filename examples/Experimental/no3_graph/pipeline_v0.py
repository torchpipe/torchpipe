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

import numpy as np
import cv2
import torch
import torchpipe as tp
from torchpipe import pipe, TASK_DATA_KEY, TASK_RESULT_KEY, TASK_BOX_KEY

tp.utils.cpp_extension.load(name="yolox", sources=["./yolox_new.cpp"])

parser = argparse.ArgumentParser()
parser.add_argument(
    "--config", dest="toml", type=str, default="./pipeline_v0.toml", help="configuration file"
)
parser.add_argument("--benchmark", action="store_true")
args = parser.parse_args()

if __name__ == "__main__":
    img_path = "../../../test/assets/norm_jpg_720p/dog.jpg"
    img = cv2.imread(img_path, 1)
    img = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])[1]
    img = img.tobytes()

    toml_path = args.toml
    print(f"toml: {toml_path}")
 
    # 调用
    model = pipe(toml_path)

    def run(img_data, save_img=False):
        img_path, img_data = img_data[0]

        ori_img = cv2.imdecode(np.frombuffer(img_data, dtype=np.uint8), cv2.IMREAD_COLOR)
        # ori_img -> img,  pad to 416 416, and get the x_ratio and y_ratio
        # use numpy and cv2
        ratio = min(416 / ori_img.shape[0], 416 / ori_img.shape[1])
        img = cv2.resize(ori_img, (int(ori_img.shape[1] * ratio), int(ori_img.shape[0] * ratio)))
        img = cv2.copyMakeBorder(img, 0, 416 - img.shape[0], 0, 416 - img.shape[1], cv2.BORDER_CONSTANT, value=(0, 0, 0))
        # float x_ratio = data.cols * 1.0f / resize_w;
        # float y_ratio = data.rows * 1.0f / resize_h;
    

        # detect
        input = {}
        input["data"] = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).cuda()
        input["ratio"] = 1.0/ratio


        # input["ratio"] = 
        input["node_name"] = "detect"

        model(input)
        
        boxes = input[TASK_BOX_KEY]
        # crop from ori_img by boxes
        # use numpy and cv2
        croped_img = []
        for box in boxes:
            x1, y1, x2, y2 = box.tolist()
            img = ori_img[int(y1):int(y2), int(x1):int(x2), :]
            img = cv2.resize(img, (224, 224))
            # cvtcolor
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # imagenet preprocess
            img = img.astype(np.float32)
            # img /= 255.0
            # img -= np.array([0.485, 0.456, 0.406])
            # img /= np.array([0.229, 0.224, 0.225])
            img = img.transpose(2, 0, 1)
            croped_img.append(torch.from_numpy(img).unsqueeze(0))


        # classify
        cls_1_inputs = [{"data":img,'node_name':'cls_1'} for img in croped_img]
        cls_2_inputs = [{"data":img,'node_name':'cls_2'} for img in croped_img]

        model(cls_1_inputs)
        cls_1_score = [x["score"] for x in cls_1_inputs]
        cls_1_class = [x["result"] for x in cls_1_inputs]
        # retry cls_1 for score < 0.3
        retry_indexes = []
        for i in range(len(cls_1_score)):
            if cls_1_score[i] < 0.3:
                retry_indexes.append(i)
        retry_cls_1_inputs = [{"data":croped_img[i],'node_name':'post_cls_1'} for i in retry_indexes]
        model(retry_cls_1_inputs)
         # update cls_1_score and cls_1_class
        for i in range(len(retry_indexes)):
            cls_1_score[retry_indexes[i]] = retry_cls_1_inputs[i]["score"]
            cls_1_class[retry_indexes[i]] = retry_cls_1_inputs[i]["result"]

        model(cls_2_inputs)
        cls_2_score = [x["score"] for x in cls_2_inputs]
        cls_2_class = [x["result"] for x in cls_2_inputs]

        

       

        if save_img:
            print("cls_1_score, cls_1_class, cls_2_score, cls_2_class: ", cls_1_score, cls_1_class, cls_2_score, cls_2_class)
            # print(input.keys())

            detect_result = input[TASK_BOX_KEY]
            # print(("detect_result: ", detect_result))

            img = cv2.imread(img_path)
            for t in range(len(detect_result)):
                x1, y1, x2, y2= detect_result[t].tolist()
                img = cv2.rectangle(
                    img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2
                )
            out_path = "dog_result.jpg"
            cv2.imwrite(out_path, img[:, :, ::1])
            print(f"result saved in: {out_path}")
        return cls_1_score, cls_1_class, cls_2_score, cls_2_class


    if args.benchmark:
        from torchpipe.utils import test

        test.test_from_raw_file(
            run,
            os.path.join("../../../test/assets/norm_jpg_720p"),
            num_clients=20,
            total_number=10000,
        )
    else:
        run([(img_path, img)], save_img=True)
