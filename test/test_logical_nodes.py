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


def test_logical_nodes(file_dir: str = "assets/norm_jpg/", num_clients=10, batch_size=1,
                       ext=[".jpg", '.JPG', '.jpeg', '.JPEG']):

    model = torchpipe.pipe("assets/toml/logical.toml")

    inputs = [{"data": 1}, {"data": 2}]

    model(inputs)
    assert (inputs[0]["result"] == inputs[0]["data.2"])
    assert (inputs[1]["result"] == inputs[1]["data.2"])


if __name__ == "__main__":
    # test_test_random(dir = "assets/norm_jpg/")

    test_logical_nodes()
