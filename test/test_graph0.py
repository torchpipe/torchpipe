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

toml_path = ["assets/toml/graph0.toml"]


class TestGraph0(object):
    @classmethod
    def setup_class(self):
        """这是一个class级别的setup函数，它会在这个测试类TestSohu里
        所有test执行之前，被调用一次.
        注意它是一个@classmethod
        """
        self.root = "r_1"
        self.config = []
        for p in toml_path:
            self.config.append(tp.parse_toml(p))

    def test_skip(self):
        config = copy.deepcopy(self.config[0])
        config["a_2"]["filter"] = "Skip"
        model = tp.pipe(config)
        input = {tp.TASK_DATA_KEY: 1, "node_name": self.root}
        model(input)
        assert input[tp.TASK_RESULT_KEY] == 1

        config["b_2"]["filter"] = "Skip"
        config["a_3"]["filter"] = "Skip"
        # config["a_3"]["map"]="b_2[result:data]"
        model = tp.pipe(config)
        input = {tp.TASK_DATA_KEY: 1, "node_name": self.root}
        model(input)
        assert input[tp.TASK_RESULT_KEY] == 1

    def test_skips(self):
        config = copy.deepcopy(self.config[0])
        config["a_2_1"]["filter"] = "Skip"
        config["r_2"]["filter"] = "Run"
        model = tp.pipe(config)
        input = {tp.TASK_DATA_KEY: "1", "node_name": self.root}
        model(input)
        assert input[tp.TASK_RESULT_KEY] == b"1"

    def test_serial_skip(self):
        config = copy.deepcopy(self.config[0])

        config["a_2_1"]["filter"] = "SerialSkip"
        config["r_2"]["filter"] = "SerialSkip"
        model = tp.pipe(config)
        input = {tp.TASK_DATA_KEY: "1", "node_name": self.root}
        model(input)
        assert tp.TASK_EVENT_KEY not in input.keys()
        assert tp.TASK_RESULT_KEY not in input.keys()

        config["r_2"]["filter"] = "Run"
        config["b_1"]["filter"] = "SerialSkip"
        config["b_1"]["backend"] = "C10Exception"
        config["b_2"]["backend"] = "C10Exception"
        model = tp.pipe(config)
        input = {tp.TASK_DATA_KEY: "1", "node_name": self.root}
        model(input)
        assert input[tp.TASK_RESULT_KEY] == b"1"
        assert tp.TASK_EVENT_KEY not in input.keys()

        config["r_2"]["filter"] = "Run"
        config["r_2"]["backend"] = "C10Exception"
        model = tp.pipe(config)
        input = {tp.TASK_DATA_KEY: "1", "node_name": self.root}
        with pytest.raises(RuntimeError):
            model(input)

    def test_graph_skip(self):
        config = copy.deepcopy(self.config[0])

        config["a_1"]["filter"] = "SubGraphSkip"
        config["a_3"]["backend"] = "C10Exception"
        # config["a_3"]["filter"]="SerialSkip"
        model = tp.pipe(config)
        input = {tp.TASK_DATA_KEY: "1", "node_name": self.root}
        model(input)
        assert input[tp.TASK_RESULT_KEY] == b"1"

        config["a_1"]["filter"] = "SubGraphSkip"
        config["r_2"]["backend"] = "C10Exception"
        # config["a_3"]["filter"]="SerialSkip"
        model = tp.pipe(config)
        input = {tp.TASK_DATA_KEY: "1", "node_name": self.root}
        with pytest.raises(RuntimeError):
            model(input)

        config["a_1"]["map"] = "[result:data,result:a_1]"
        config["a_1"]["filter"] = "SubGraphSkip"
        config["r_2"]["map"] = "a_3[a_1:data], b_2[result:b_2]"
        config["r_2"]["backend"] = "Identity"
        model = tp.pipe(config)
        input = {tp.TASK_DATA_KEY: "1", "node_name": self.root}
        model(input)
        assert input["b_2"] == b"1"
        assert input[tp.TASK_RESULT_KEY] == b"1"
        # with pytest.raises(IndexError):
        #     model(input)

    def test_stop(self):
        config = copy.deepcopy(self.config[0])

        config["a_1"]["map"] = "[result:data,result:a_1]"
        config["a_2_1"]["filter"] = "Break"
        config["a_2_1"]["map"] = "a_2[a_1:data,a_1:a_2_1]"
        config["r_2"]["backend"] = "C10Exception"
        model = tp.pipe(config)
        input = {tp.TASK_DATA_KEY: "1", "node_name": self.root}
        model(input)
        assert input["a_2_1"] == b"1"
        assert tp.TASK_RESULT_KEY not in input.keys()


if __name__ == "__main__":
    import time

    time.sleep(5)
    a = TestGraph0()
    a.setup_class()

    a.test_skip()
