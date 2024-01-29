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

import os
import time
import torchpipe
# import torchpipe.utils.test
from typing import List


import pytest


def test_multiple_inputs(file_dir: str = "assets/norm_jpg/", num_clients=10, batch_size=1,
                         ext=[".jpg", '.JPG', '.jpeg', '.JPEG']):

    model = torchpipe.pipe({"backend": "DecodeTensor"})
    file_broken = "assets/hws_jpg/empty.jpg"

    file_norm = "assets/norm_jpg/dog.jpg"

    file_broken_bytes = open(file_broken, "rb").read()
    file_norm = open(file_norm, "rb").read()
    inputs = [{"data": file_broken_bytes}, {"data": file_norm}]

    model(inputs)
    assert (inputs[1]["result"].shape == (1, 3, 576, 768))
    assert ("result" not in inputs[0].keys())


def test_multiple_inputs_except(file_dir: str = "assets/norm_jpg/", num_clients=10, batch_size=1,
                                ext=[".jpg", '.JPG', '.jpeg', '.JPEG']):

    model = torchpipe.pipe({"backend": "DecodeMat"})
    file_broken = "assets/hws_jpg/empty.jpg"

    file_norm = "assets/norm_jpg/dog.jpg"

    file_broken_bytes = open(file_broken, "rb").read()
    file_norm = open(file_norm, "rb").read()
    inputs = [{"data": file_broken_bytes}, {"data": file_norm}]

    with pytest.raises(RuntimeError):
        model(inputs)


if __name__ == "__main__":
    # test_test_random(dir = "assets/norm_jpg/")

    test_multiple_inputs_except()
