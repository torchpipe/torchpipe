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
from typing import List

import torch
import pytest


def test_multiple_inputs(file_dir: str = "assets/norm_jpg/", num_clients=10, batch_size=1,
                         ext=[".jpg", '.JPG', '.jpeg', '.JPEG']):

    model = torchpipe.pipe({"backend": "BorrowReplay", "max_batch_size": 123})
    data = torch.randn((3, 3, 576, 768))
    inputs = {"data": [data], "borrow_type": "borrow_or_insert", "id":1}
    model(inputs)
    
    data_center = torch.randn((119, 3, 576, 768))
    inputs = {"data": [data_center], "borrow_type": "borrow_or_insert", "id":2}
    model(inputs)

    data = torch.zeros((2, 3, 576, 768))
    
    inputs = {"data": [data], "borrow_type": "borrow_or_insert", "id":3}

    model(inputs)
    assert(inputs["result"][1][0].shape[0] == 118)
    assert(torch.equal(inputs["result"][1][0][:,...], data_center[:118, ...]))



def test_force_batch():

    model = torchpipe.pipe({"backend": "BorrowReplay", "max_batch_size": 123})
    data = torch.randn((3, 3, 576, 768))
    inputs = {"data": [data], "borrow_type": "borrow_or_insert", "id":1}
    model(inputs)
    
    data_center = torch.randn((111, 3, 576, 768))
    inputs = {"data": [data_center], "borrow_type": "borrow_or_insert", "id":2}
    model(inputs)

    data = [torch.randn((2, 3, 576, 768))]
    
    inputs = {"data": data, "borrow_type": "borrow_all", "id":3}

    model(inputs)
    assert(inputs["result"][1][0].shape[0] == 111)
    assert(torch.equal(inputs["result"][1][0][:,...], data_center[:111, ...]))


    data = torch.randn((2, 3, 576, 768))
    data_center = torch.randn((111, 3, 576, 768))
    
    inputs = {"data": [[data], [data_center]], "borrow_type": "set_replay", "id":3}
    model(inputs)

    inputs = {"data": [], "borrow_type": "get_replay", "id":2}
    model(inputs)
    # print(inputs["result"])
    assert(torch.equal(inputs["result"][0][0], data_center ))

def test_multiple_input():

    

    model = torchpipe.pipe({"backend": "BorrowReplay", "max_batch_size": 123})
    data = [torch.randn((3, 3, 576, 768)), torch.randn((3, 3, 576, 768))]
    inputs = {"data": data, "borrow_type": "borrow_or_insert", "id":1}
    model(inputs)
    
    
    
    data_center = [torch.randn((111, 3, 576, 768)), torch.randn((111, 3, 576, 768))]
    inputs = {"data": data_center, "borrow_type": "borrow_or_insert", "id":2}
    model(inputs)
     

    data =[torch.randn((2, 3, 576, 768)),torch.randn((2, 3, 576, 768))] 
    
    inputs = {"data": data, "borrow_type": "borrow_all", "id":3}

    model(inputs)
    assert(inputs["result"][1][0].shape[0] == 111)
    assert(torch.equal(inputs["result"][1][0][:,...], data_center[0][:111, ...]))

    

    data = torch.randn((2, 3, 576, 768))
    data_center = torch.randn((111, 3, 576, 768))
    
    inputs = {"data": [[data], [data_center]], "borrow_type": "set_replay", "id":3}
    model(inputs)

    inputs = {"data": [], "borrow_type": "get_replay", "id":2}
    model(inputs)
    # print(inputs["result"])
    assert(torch.equal(inputs["result"][0][0], data_center ))

if __name__ == "__main__":
    import time
    # time.sleep(5)
    # test_test_random(dir = "assets/norm_jpg/")

    test_force_batch()
