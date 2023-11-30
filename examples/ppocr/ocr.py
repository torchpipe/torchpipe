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

import os
import glob
import cv2
import logging
import torch

from PIL import Image
import numpy as np
from draw_ocr import draw_ocr_box_txt


from torchpipe.utils import cpp_extension
import torchpipe as tp

cwd = os.path.dirname(os.path.abspath(__file__))
files = glob.glob(os.path.join(cwd, "csrc/src/postprocess_op.cpp"))
files += glob.glob(os.path.join(cwd, "csrc/ocr.cpp"))
files += glob.glob(os.path.join(cwd, "csrc/src/utility.cpp"))


print("compile: ", files)
cpp_extension.load(name="dbnet", sources=files,
                   extra_include_paths=[os.path.join(cwd, "csrc/"),
                                        os.path.join(cwd, "ppocr/cpp/"), "/usr/local/include/opencv4/"],
                   extra_ldflags=["-lopencv_core", "-lopencv_imgcodecs", "-lopencv_imgproc"], rebuild_if_exist=True, is_python_module=False)


logging.basicConfig(level=logging.DEBUG)


if __name__ == "__main__":

    # prepare data:
    img_path = "../../test/assets/image/lite_demo.png"
    # img_path = "./ppocr/lite_demo.png"
    gimg = cv2.imread(img_path, 1)

    import time
    # time.sleep(5)

    # 调用
    from torchpipe import pipe, TASK_DATA_KEY, TASK_RESULT_KEY, TASK_BOX_KEY
    nodes = pipe("ocr.toml")
    for i in range(1*1):
        input = {}

        input[TASK_DATA_KEY] = torch.from_numpy(
            gimg).permute(2, 0, 1).unsqueeze(0)
        input["color"] = "bgr"
        input["node_name"] = "jpg_decoder"

        nodes(input)
    # for i in range(len(input[TASK_RESULT_KEY])):
    #     print(input[TASK_RESULT_KEY][i].shape, input[TASK_RESULT_KEY][i].shape[1]/input[TASK_RESULT_KEY][i].shape[0])
    print(input.keys())
    if input[TASK_RESULT_KEY]:
        print(type(input[TASK_RESULT_KEY][0]))
        try:
            print("_"*10+"\n此处打印用于检测python是否支持输出中文。 \n"
                  + "_"*10)
        except:
            print(
                "chinese not supported. try: \n LANG=zh.CN.utf8 \n source /etc/profile \n")
        # print(len(input[TASK_RESULT_KEY]), input[TASK_RESULT_KEY][0].shape)
        # exit(0)
        # print(input[TASK_RESULT_KEY][0].encode(
        #     "unicode_escape").decode("unicode-escape"))
        print(input[TASK_RESULT_KEY][0].decode('utf8'))

        # print((input[TASK_RESULT_KEY][0]), input[TASK_RESULT_KEY])
        boxes = input[TASK_BOX_KEY]
        scores = input["scores"]
        texts = []
        for text in input[TASK_RESULT_KEY]:
            texts.append(text.decode("utf8"))
        image = Image.fromarray(cv2.cvtColor(gimg, cv2.COLOR_BGR2RGB))

        draw_img = draw_ocr_box_txt(
            image, boxes, texts, scores, show_score=True, drop_score=0.5)
        draw_img_save_dir = "./"
        # os.makedirs(draw_img_save_dir, exist_ok=True)

        image_file = "ocr_vis_py.png"
        cv2.imwrite(
            os.path.join(draw_img_save_dir, os.path.basename(image_file)),
            draw_img[:, :, ::-1])
        print("The visualized image saved in {}".format(
            os.path.join(draw_img_save_dir, os.path.basename(image_file))))
    else:
        boxes = input["_box"]
        scores = input["scores"]

        print(boxes, scores, input[TASK_RESULT_KEY])

    def run(img_input):
        img_path, img_raw = img_input[0]
        nparr = np.fromstring(img_raw, np.uint8)
        img_data = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # print("decode result: ", input[TASK_RESULT_KEY].shape, input["color"])
        input = {}

        input[TASK_DATA_KEY] = torch.from_numpy(
            gimg).cuda().permute(2, 0, 1).unsqueeze(0)
        input["node_name"] = "jpg_decoder"
        input["color"] = "bgr"
        nodes(input)
        # return None
        # print(input[TASK_RESULT_KEY][0].shape)
        return input[TASK_RESULT_KEY][0][0]
        # or nodes("resnet18",  input)

        # print("resnet18 result: ", len(input[TASK_RESULT_KEY]), input[TASK_RESULT_KEY][0].shape)

    # run([(img_path, img)])
    from torchpipe.tool import test_tools
    test_tools.test_from_raw_jpg(run, os.path.join( "./"))
    # c1    2200 b4     1181 b1
    # c2    2122 b4     1650 b1
    # c4    2124 b4     1976 b1
