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
import cv2
import os
import torchpipe as tp
import copy

import pytest

toml_path = ["assets/toml/graph2.toml"]


class TestGraph1(object):
    @classmethod
    def setup_class(self):
        """ 这是一个class级别的setup函数，它会在这个测试类TestSohu里
        所有test执行之前，被调用一次.
        注意它是一个@classmethod
        """
        self.root = "det"
        self.config = []
        for p in toml_path:
            self.config.append(tp.parse_toml(p))

    def test_skip(self):
        config = copy.deepcopy(self.config[0])
        # config["a_2"]["filter"]="Skip"
        model = tp.pipe(config)
        input = {tp.TASK_DATA_KEY: 1, "node_name": self.root}
        model(input)
        print(input)
        assert (len(input[tp.TASK_RESULT_KEY]) ==
                5 and input[tp.TASK_RESULT_KEY][0] == 1)

        # config["b_2"]["filter"]="Skip"
        # config["a_3"]["filter"]="Skip"
        # # config["a_3"]["map"]="b_2[result:data]"
        # model = tp.pipe(config)
        # input={tp.TASK_DATA_KEY:1, "node_name":self.root}
        # model(input)
        # assert(input[tp.TASK_RESULT_KEY] == 1)
    def test_jump_exception(self):
        config = copy.deepcopy(self.config[0])
        config["rec.2"]["backend"] = "C10Exception"
        model = tp.pipe(config)
        input = {tp.TASK_DATA_KEY: 1, "node_name": self.root}
        model(input)
        print(input)
        assert (len(input[tp.TASK_RESULT_KEY]) ==
                5 and input[tp.TASK_RESULT_KEY][0] == 1)


if __name__ == "__main__":
    import time
    # time.sleep(10)
    a = TestGraph1()
    a.setup_class()
    for i in range(100):
        a.test_jump_exception()
