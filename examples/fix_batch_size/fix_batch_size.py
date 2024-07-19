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
import io
import os
import torchpipe
from torchpipe import pipe, TASK_DATA_KEY, TASK_RESULT_KEY, TASK_INFO_KEY
import tempfile
import numpy as np


class MultiIdentity(torch.nn.Module):
    def __init__(self):
        super(MultiIdentity, self).__init__()
        self.identity = torch.nn.Identity()

    def forward(self, x_and_y):
        x, y = x_and_y
        x = self.identity(x)
        return x, x+y


if __name__ == "__main__":
    start_beta = "b1"  # avoid being replaced
    if (torchpipe.__version__ < "0.2.2"+start_beta):
        print("this example requires torchpipe >= 0.2.2"+start_beta)
        exit(0)
    import time
    # time.sleep(6)

    identity_model = MultiIdentity().eval()

    data_bchw = torch.rand((1, 3, 224, 282))
    onnx_path_1 = os.path.join(tempfile.gettempdir(), "Identity_1.onnx")
    print("export: ", onnx_path_1)
    torch.onnx.export(identity_model, [data_bchw, data_bchw], onnx_path_1,
                      opset_version=17,
                      do_constant_folding=True,
                      input_names=["input", "inputB"], output_names=["output", "outputB"])

    data_bchw = torch.rand((4, 3, 224, 282))
    onnx_path_4 = os.path.join(tempfile.gettempdir(), "Identity_4.onnx")
    print("export: ", onnx_path_4)
    torch.onnx.export(identity_model, [data_bchw, data_bchw], onnx_path_4,
                      opset_version=17,
                      do_constant_folding=True,
                      input_names=["input", "inputB"], output_names=["output", "outputB"])

    # prepare data:
    import cv2
    img_path = "../../test/assets/encode_jpeg/grace_hopper_517x606.jpg"
    img_out = cv2.imread(img_path, 1)
    img_out = cv2.imencode(
        '.jpg', img_out, [int(cv2.IMWRITE_JPEG_QUALITY), 95])[1]

    img_out = img_out.tobytes()

    model = torchpipe.pipe({"model": onnx_path_1+"&"+onnx_path_4, "backend": "SyncTensor[TensorrtTensor]",
                            'instance_num': 2, 'batching_timeout': '2', 'min': "1&4", 'max': "1&4"})

    data = torch.zeros((224, 282, 3), dtype=torch.int8)
    # or
    data = torch.zeros((3, 224, 282), dtype=torch.int8)
    input = {"data": [data, data]}

    def run(img):
        img_path, img = img[0]
        img = np.frombuffer(img, dtype=np.uint8)
        img = cv2.resize(cv2.imdecode(img, cv2.IMREAD_COLOR), (282, 224))
        img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)

        # 只有一个节点，node_name默认产生 ; img.clone()
        input = {TASK_DATA_KEY: [img, img]}

        model(input)

        if TASK_RESULT_KEY not in input.keys():
            print("error decode")
            return

        return input[TASK_RESULT_KEY][0], input[TASK_RESULT_KEY][1]

    # # 简单测试：
    run([(img_path, img_out)])

    # # 异步压测：
    from torchpipe.utils.test import test_from_raw_file
    test_from_raw_file(run, os.path.join(
        "../..", "test/assets/encode_jpeg/"), total_number=10000)
