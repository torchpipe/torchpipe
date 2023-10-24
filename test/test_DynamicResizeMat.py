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


import io
from collections import OrderedDict
from typing import List, Tuple

import os
import pytest
import cv2
import numpy as np
import torch
import torchpipe


class TestOpenCVResize:
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
    # src_array = src_arrays[0]

    @classmethod
    def setup_class(self):
        pass

    @pytest.mark.parametrize("target_h,target_w", [(1, 224), (224, 1), (224, 224), (1, 1), (43, 99), (2, 2)])
    @pytest.mark.parametrize("src_array", src_arrays)
    def test_DynamicResizeMat(self, target_h, target_w, src_array):
        resized_img = cv2.resize(src_array,
                                 (target_w, target_h))
        img = resized_img
        print(img.shape)

        imnpoit = {"data": torch.from_numpy(src_array), "resize_h": (
            target_h), "resize_w": (target_w)}

        pipe_tensor = torchpipe.pipe(
            {"a": {'backend': "Sequential[Tensor2Mat,Parallel[DynamicResizeMat]]"}})
        pipe_tensor(imnpoit)
        print(imnpoit['result'])
        print(imnpoit['result'].shape)
        # .cpu().numpy()#.squeeze(0).permute(1, 2, 0).cpu().numpy()
        img_pipe = imnpoit['result']

        zzz = img_pipe-img  # .astype(np.float32)
        if img_pipe.shape[0] > 10:
            print("xxx",  np.max(np.fabs(zzz)), "\n")
            print(np.where(np.fabs(zzz) == 1), "\n")
            # print("pipe=", img_pipe[0, 11, :], "opencv=", img[0, 11, :])
        else:
            print(zzz, "xxx pipe=", img_pipe, " opencv=", img)
        # print(np.max(img_pipe), np.max(np.fabs(zzz)))
        # print(img[223:,222:,0:],"\n",img_pipe[223:,222:,0:])
        # print(img[223:,222:,0:],"\n",img_pipe[223:,222:,0:])

        assert (np.max(np.fabs(zzz)) <= 1)


if __name__ == "__main__":
    import time
    time.sleep(5)

    a = TestOpenCVResize()
    a.setup_class()
    # img= cv2.imread("assets/image/lite_demo.png")
    if True:
        a.test_DynamicResizeMat(2, 2, a.src_arrays[0])
    else:
        a.test_infer(224, 224, a.src_arrays[0])
