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


class TestAlignMat:
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

    @pytest.mark.parametrize("target_h,max_w", [(1, 224), (224, 1), (224, 224), (1, 1), (43, 99), (2, 2)])
    @pytest.mark.parametrize("src_array", src_arrays)
    @pytest.mark.parametrize("align", [1, 3, 32])
    def test_AlignMat(self, target_h, max_w, src_array, align):

        if max_w % align != 0:
            with pytest.raises(RuntimeError):
                torchpipe.pipe({"resize_h": target_h, "max_w": max_w, "align": align,
                                'backend': "Sequential[Tensor2Mat, AlignMat]"})
            return

        h = src_array.shape[0]
        w = src_array.shape[1]

        imnpoit = {"data": torch.from_numpy(src_array)}

        pipe_tensor = torchpipe.pipe({"resize_h": target_h, "max_w": max_w, "align": align,
                                      'backend': "Sequential[Tensor2Mat, AlignMat]"})
        pipe_tensor(imnpoit)
        assert (imnpoit['resize_h'] == target_h)

        target_w = int(target_h * w / float(h))
        if target_w % align != 0:
            target_w = (target_w//align+1)*align
        if target_w > max_w:
            target_w = max_w
        assert (target_w == imnpoit['resize_w'])

        # img_pipe = imnpoit['result'].cpu().numpy()#.squeeze(0).permute(1, 2, 0).cpu().numpy()

    def test_AlignMats(self):
        target_h, max_w, src_array, align = 12, 22, self.src_arrays[-1], 2

        h = src_array.shape[0]
        w = src_array.shape[1]
        print(h, w)

        imnpoit = [{"data": torch.from_numpy(src_array)}, {
            "data": torch.from_numpy(src_array[:, :14, :])}]

        pipe_tensor = torchpipe.pipe({"resize_h": target_h, "max_w": max_w, "align": align,
                                      'backend': "Sequential[Tensor2Mat, Parallel[AlignMat]]"})
        pipe_tensor(imnpoit)
        assert (imnpoit[1]['resize_h'] == target_h)
        assert (imnpoit[0]['resize_h'] == target_h)

        target_w = int(target_h * w / float(h))
        if target_w % align != 0:
            target_w = (target_w//align+1)*align
        if target_w > max_w:
            target_w = max_w
        assert (target_w == imnpoit[1]['resize_w'])
        assert (target_w == imnpoit[0]['resize_w'])


if __name__ == "__main__":
    import time
    # time.sleep(5)

    a = TestAlignMat()
    a.setup_class()
    a.test_AlignMat()
    exit(0)
    # img= cv2.imread("assets/image/lite_demo.png")
    if True:
        a.test_AlignMat(2, 2, a.src_arrays[3], 32)
    else:
        a.test_infer(224, 224, a.src_arrays[0])
