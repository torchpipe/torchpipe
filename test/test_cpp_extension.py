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

from torchpipe.utils.cpp_extension import load
from torchpipe import pipe
import torchpipe as tp
import torch
import numpy

data_map = {
    "data": "",
    "str": "123",
    "bytes": "123".encode("utf-8"),
    "torch.Tensor": torch.zeros(1),
    "bool": True,
    "float": 0.123,
    "int": 123,
}
load(
    sources=["csrc/PY2CPP.cpp", "csrc/CPP2PY.cpp"],
    extra_include_paths=["/usr/local/include/opencv4/"],
)


def test_load():
    # 加载c++
    load(sources=["csrc/CustomIdentity.cpp"])

    # 初始化
    model = pipe({"backend": "CustomIdentity", "instance_num": "2"})

    # 前向
    input = {"data": "123"}
    model(input)
    # 检查结果
    assert str(input["result"], encoding="utf-8") == "123"


def test_PY2CPP():
    # 加载c++
    # 初始化
    model = pipe({"backend": "PY2CPP", "instance_num": "2"})

    # 前向
    input = data_map
    model(input)


def test_ListPY2CPP():
    # 加载c++

    # 初始化
    model = pipe({"backend": "ListPY2CPP", "instance_num": "2"})

    # 前向
    input = {key: [value] * 3 for key, value in data_map.items()}
    model(input)

    input = {key: [] for key, value in data_map.items()}
    model(input)


def test_SetPY2CPP():
    # 加载c++

    # 初始化
    model = pipe({"backend": "SetPY2CPP", "instance_num": "2"})

    input = {
        key: {value, value} for key, value in data_map.items() if key != "torch.Tensor"
    }
    model(input)
    input = {key: set()
             for key, value in data_map.items() if key != "torch.Tensor"}
    model(input)


def test_StrMapPY2CPP():
    # 加载c++

    # 初始化
    model = pipe({"backend": "StrMapPY2CPP", "instance_num": "2"})

    input = {
        key: {key: value, key: value} for key, value in data_map.items() if key != ""
    }
    model(input)
    input = {key: dict() for key, value in data_map.items() if key != ""}
    model(input)


def test_CPP2PY():
    # 加载c++
    # 初始化
    model = pipe({"backend": "CPP2PY", "instance_num": "2"})

    # 前向
    input = data_map
    model(input)
    # assert type(input["any"]) == tp.any


def test_ListCPP2PY():
    # 加载c++

    # 初始化
    model = pipe({"backend": "ListCPP2PY", "instance_num": "2"})

    # 前向
    input = {key: [] for key, value in data_map.items()}
    model(input)
    print(input)
    assert type(input["std::string"]) == list
    assert (input["at::Tensor"][0]) is not None
    assert input["any"][0] == 1
    assert type(input["any"][1]) == list
    assert type(input["any"][1][0]) == torch.Tensor


def test_List2CPP2PY():
    # 加载c++

    # 初始化
    model = pipe({"backend": "List2CPP2PY", "instance_num": "2"})

    # 前向
    input = {key: [] for key, value in data_map.items()}
    model(input)
    print(input)
    assert type(input["double"]) == list
    assert type(input["double"][0]) == numpy.ndarray
    assert len(input["empty"]) == 0


def test_SetCPP2PY():
    model = pipe({"backend": "SetCPP2PY", "instance_num": "2"})

    input = {key: set() for key, value in data_map.items()}
    model(input)
    print(input)


def test_StrMapCPP2PY():
    model = pipe({"backend": "StrMapCPP2PY", "instance_num": "2"})

    input = {key: {key: ""} for key, value in data_map.items()}
    model(input)
    print(input)

    assert input["any_str_dict"]["int"] == 1
    assert type(input["any_str_dict"]["at::Tensor"]) == list
    assert type(input["any_str_dict"]["at::Tensor"][1]) == torch.Tensor


# def test_StrMapPY2CPP():

#     ## 加载c++


#     ## 初始化
#     model = pipe({"backend":"StrMapPY2CPP", "instance_num":"2"})


#     input = {key:{key:value,key:value}  for key,value  in data_map.items() if key != ""}
#     model(input)
#     input = {key:dict()  for key,value  in data_map.items() if key != ""}
#     model(input)


def test_import_all():
    from torchpipe.utils import test, cpp_extension


def test_import():
    import torchpipe.utils as utils_tmp
    from torchpipe import utils

    a = utils.cpp_extension
    b = utils.cpp_extension.load

    from torchpipe.utils import cpp_extension


if __name__ == "__main__":
    import time

    # time.sleep(5)
    test_CPP2PY()
    # test_StrMapCPP2PY()
