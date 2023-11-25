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
import time
import torchpipe
# import torchpipe.utils.test
from typing import List,Dict,Any

input = list(range(0, 10000))
result = {}

class AnyPythonBackend:
    def init(self, config_param: Dict[str, str]) -> None:
        print("config_param",config_param)
        self.add = int(config_param["config"])

    def forward(self, data: [Dict[str, Any]]) -> None:
        data[0]["result"] = data[0]["data"] + self.add


class TestBackend:
    @classmethod
    def setup_class(self):

        torchpipe.register_backend(AnyPythonBackend, "py_model_1")
        self.model_trt = torchpipe.pipe(
            {"backend": "Python[py_model_1]","config":1,"any_key_u_want":"uhiojh"})
 
    def test_register_py(self):
        input = {"data":2}
        self.model_trt([input])
        print(input["result"])
        assert input["result"] == 3
        
        
 
if __name__ == "__main__":
    # test_test_random(dir = "assets/norm_jpg/")

    # test_all_files(file_dir = "../examples/ocr_poly_v2/test_img/", num_clients=10)
    a = TestBackend()
    a.setup_class()
    a.test_register_py()
