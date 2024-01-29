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


import io
from collections import OrderedDict
from typing import List, Tuple

import os
import pytest
import cv2
import numpy as np
import torch
import torchpipe


class TestMat2Tensor:
    src_arrays = []
    src_arrays.append(np.array([[[255, 255, 254],
                                 [0,   1,   0],
                                 [47,  48,  46]],

                                [[0,   1,   0],
                                 [59,  60,  58],
                                 [7,   8,   6]],

                                [[49,  50,  48],
                                 [0,   1,   0],
                                 [255, 255, 254]]], dtype=np.uint8))
    src_arrays.append(np.array([[[0, 0, 0], [1,   0, 0],
                                 [1,  0,  0]]],
                               dtype=np.uint8))
    src_arrays.append(
        np.array([[[0, 0, 0]], [[1, 0, 0]], [[1, 0, 0]]], dtype=np.uint8))
    src_arrays.append(cv2.imread("assets/image/lite_demo.png"))
    ori_len = len(src_arrays)
    for i in range(ori_len):
        src_arrays.append(src_arrays[i].astype(np.float32))

    # src_array = src_arrays[0]

    @classmethod
    def setup_class(self):
        pass

        # ,Tensor2Mat, SaveMat,Mat2Tensor

        # self.pipe = torchpipe.pipe({"a": {"resize_h": target_h, "resize_w": target_w, "color": 'bgr',
        #                                   'Interpreter::backend': "Sequential[Tensor2Mat,cvtColorMat,OpenCVResizeMat,Mat2Tensor,SyncTensor]"}})
        # self.pipe_tensor = torchpipe.pipe({"a": {"resize_h": target_h, "resize_w": target_w, "color": 'bgr',
        #                                          "save_dir": "./",
        #                                          'Interpreter::backend': "Sequential[Tensor2Mat,cvtColorMat,Mat2Tensor,OpenCVResizeTensor, SyncTensor]"}})

    # @pytest.mark.parametrize("target_h,target_w", [(1, 224),(224, 1),(224, 224), (1, 1), (43, 99)])
    # @pytest.mark.parametrize("src_array", src_arrays)
    # def test_infer(self, target_h,target_w, src_array):
    #     # print(self.src_array.shape)
    #     opencv_img = cv2.resize(src_array,(target_w, target_h))
    #     img = opencv_img
    #     # print(img.shape)
    #     # print(np.asarray(opencv_img))
    #     imnpoit = {"data": torch.from_numpy(src_array), 'color': "bgr"}

    #     pipe = torchpipe.pipe({"a": {"resize_h": target_h, "resize_w": target_w,
    #                                       'Interpreter::backend': "SyncTensor[OpenCVResizeTensor]"}})
    #     pipe(imnpoit)
    #     img_pipe = imnpoit['result'].squeeze(0).permute(1, 2, 0).cpu().numpy()
    #     zzz = img_pipe-img
    #     print(zzz, "xx")
    #     assert (np.max(np.fabs(zzz)) == 0)
    #     # print(img[223:,222:,0:])


    def test_ReturnTensor(self):
        # resized_img = cv2.resize(src_array,
        #     (target_w, target_h))
        # img = resized_img
        import time
        # print(src_array, src_array.shape)
        # time.sleep(10)
        with open("assets/norm_jpg/dog.jpg", "rb") as f:
            data = f.read()
        imnpoit = {"data": data}
        # print("t")

        pipe_tensor = torchpipe.pipe({"a": {"resize_h": 1, "data_format": "hwc", "color": 'bgr',
                                            'backend': "Sequential[DecodeMat,Mat2Tensor,SyncTensor]"}})

        pipe_tensor(imnpoit)
        print(imnpoit["result"].shape)
        assert ((576, 768, 3) == imnpoit["result"].shape)
        assert((imnpoit["data_format"].decode('utf-8')) == "hwc")

        pipe_tensor = torchpipe.pipe({"a": {"resize_h": 1, "data_format": "nchw", "color": 'bgr',
                                            'backend': "Sequential[DecodeMat,Mat2Tensor,SyncTensor]"}})

        pipe_tensor(imnpoit)
        print(imnpoit["result"].shape)
        assert ((1,3,576, 768) == imnpoit["result"].shape)
        


if __name__ == "__main__":
    import time
    time.sleep(5)

    a = TestMat2Tensor()
    a.setup_class()
    a.test_ReturnTensor()