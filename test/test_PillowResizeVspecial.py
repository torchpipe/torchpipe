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
import PIL
import torchpipe
from PIL import Image
if not hasattr(PIL.Image, 'Resampling'):  # Pillow<9.0
    PIL.Image.Resampling = PIL.Image


class TestPillowResize:
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

    # src_arrays.append(torch.load("assets/image/cpu_posdecoder_140008736413440_0.pt").squeeze(0).permute(1,2,0).cpu().numpy())
    src_arrays.append(cv2.imread("assets/image/pillow_diff_same.png"))
    # src_array = src_arrays[0]

    @classmethod
    def setup_class(self):
        pass

        # ,Tensor2Mat, SaveMat,Mat2Tensor

        # self.pipe = torchpipe.pipe({"a": {"resize_h": target_h, "resize_w": target_w, "color": 'bgr',
        #                                   'Interpreter::backend': "Sequential[Tensor2Mat,cvtColorMat,PillowResizeMat,Mat2Tensor,SyncTensor]"}})
        # self.pipe_tensor = torchpipe.pipe({"a": {"resize_h": target_h, "resize_w": target_w, "color": 'bgr',
        #                                          "save_dir": "./",
        #                                          'Interpreter::backend': "Sequential[Tensor2Mat,cvtColorMat,Mat2Tensor,PillowResizeTensor, SyncTensor]"}})

    @pytest.mark.parametrize("target_h,target_w", [(1, 224), (224, 1), (224, 224), (1, 1), (43, 99), (2, 2)])
    @pytest.mark.parametrize("src_array", src_arrays)
    def test_infer(self, target_h, target_w, src_array):
        # print(self.src_array.shape)
        pil_img_ori = Image.fromarray(src_array)
        pil_img = pil_img_ori.resize(
            (target_w, target_h), Image.Resampling.BILINEAR)
        img = np.asarray(pil_img)
        # print(img.shape)
        # print(np.asarray(pil_img))
        imnpoit = {"data": torch.from_numpy(
            np.asarray(pil_img_ori)), 'color': "bgr"}

        pipe = torchpipe.pipe({"a": {"resize_h": target_h, "resize_w": target_w, "color": 'bgr',
                                     'Interpreter::backend': "Sequential[Tensor2Mat,cvtColorMat,PillowResizeMat,Mat2Tensor,SyncTensor]"}})
        pipe(imnpoit)
        img_pipe = imnpoit['result'].squeeze(0).permute(1, 2, 0).cpu().numpy()
        zzz = img_pipe-img
        print(zzz, "xx")
        assert (np.max(np.fabs(zzz)) == 0)
        # print(img[223:,222:,0:])

    @pytest.mark.parametrize("target_h,target_w", [(1, 224), (224, 1), (224, 224), (1, 1), (43, 99), (2, 2)])
    @pytest.mark.parametrize("src_array", src_arrays)
    def test_PillowResizeTensor(self, target_h, target_w, src_array):
        pil_img_ori = Image.fromarray(src_array)
        pil_img = pil_img_ori.resize(
            (target_w, target_h), Image.Resampling.BILINEAR)
        img = np.asarray(pil_img)
        # print(img.shape)
        # print(np.asarray(pil_img))
        imnpoit = {"data": torch.from_numpy(
            np.asarray(pil_img_ori)), 'color': "bgr"}

        pipe_tensor = torchpipe.pipe({"a": {"resize_h": target_h, "resize_w": target_w, "color": 'bgr',
                                            "save_dir": "./",
                                            'Interpreter::backend': "Sequential[Tensor2Mat,cvtColorMat,Mat2Tensor,PillowResizeTensor, SyncTensor]"}})
        pipe_tensor(imnpoit)
        print("result= ", imnpoit['result'].shape)
        img_pipe = imnpoit['result'].squeeze(0).permute(1, 2, 0).cpu().numpy()
        zzz = img_pipe-img.astype(np.float32)
        if img_pipe.shape[0] > 10:
            print("xxx",  np.max(np.fabs(zzz)))
            print(np.where(np.fabs(zzz) == 1))
            # print("pipe=", img_pipe[7, 160, :], "pil=", img[7, 160, :], "ori=", src_array[7, 160, :])
        else:
            print("diff=", zzz, "xxx pipe=", img_pipe,
                  " pil=", img, "ori=", src_array)
        # print(np.max(img_pipe), np.max(np.fabs(zzz)))
        # print(img[223:,222:,0:],"\n",img_pipe[223:,222:,0:])
        # print(img[223:,222:,0:],"\n",img_pipe[223:,222:,0:])

        assert (np.max(np.fabs(zzz)) == 0)


if __name__ == "__main__":
    import time
    # time.sleep(5)

    a = TestPillowResize()
    a.setup_class()
    src = a.src_arrays[len(a.src_arrays)-1]  # [10:11,160:161,:]

    # print(src, "xxx",src.shape)
    # cv2.imwrite("assets/image/pillow_diff_same.png", src)
    if True:
        a.test_PillowResizeTensor(1, 1, src.copy())
    else:
        a.test_infer(224, 224, a.src_arrays[0])
