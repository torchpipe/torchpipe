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


from typing import (
    List,
    Tuple
)
import random, os

class Sampler:
    def __init__(self):
        pass
    def __call__(self, start_index: int) -> None:
        raise NotImplementedError

    def batchsize(self):
        return 1


class RandomSampler(Sampler):
    def __init__(self, data_source: List, batch_size=1):
        super().__init__()
        self.data_source = data_source
        self.batch_size = batch_size
        assert(batch_size>0)
        
        assert(0 < len(data_source))
        for i in range(batch_size):
            if len(data_source) < batch_size:
                data_source.append(data_source[i])

    def __call__(self, start_index: int):
        data = random.sample(self.data_source, self.batch_size)
        self.forward(data)

    def forward(self, data: List):
        raise RuntimeError("Requires users to implement this function")

    def batchsize(self):
        return self.batch_size

class SequentialSampler(Sampler):
    def __init__(self, data: List, batch_size=1):
        super().__init__()
        self.data = data
        self.batch_size = batch_size
        assert(len(data) >= batch_size)

    def __call__(self, start_index: int) -> None:
        data = self.data[start_index: start_index+self.batch_size]
        self.forward(data)

    def batchsize(self):
        return self.batch_size

    def forward(self, data: List):
        raise RuntimeError("Requires users to implement this function")

class LoopSampler(Sampler):
    def __init__(self, data: List, batch_size=1):
        super().__init__()
        self.data = data
        self.batch_size = batch_size
        assert(len(data) >= batch_size)
        self.length = len(data) - batch_size + 1
        for i in range(batch_size):
            self.data.append(data[i])

    def __call__(self, start_index: int) -> None:
        start_index = start_index%(self.length)
        data = self.data[start_index: start_index+self.batch_size]
        self.forward(data)

    def batchsize(self):
        return self.batch_size

    def forward(self, data: List):
        raise RuntimeError("Requires users to implement this function")


class FileSampler(LoopSampler):
    def __init__(self, data: List, batch_size=1):
        super().__init__(data, batch_size)
        self.local_result = {}

    def forward(self, data: List):
        raw_bytes = []
        for file_path in data:
            with open(file_path, "rb") as f:
                raw_bytes.append((file_path, f.read()))
        self.handle_data(raw_bytes)

    def handle_data(self, raw_bytes):
        raise RuntimeError("Requires users to implement this function")


def preload(file_dir, num_preload = 1000, recursive=True, ext=[".jpg", '.JPG', '.jpeg', '.JPEG']) -> List[Tuple[str, bytes]]:
    if not os.path.exists(file_dir):
        raise RuntimeError(file_dir+" not exists")


    list_images = []
    result = []
    if recursive:
        for root, folders, filenames in os.walk(file_dir):
            for filename in filenames:
                if os.path.splitext(filename)[-1] in ext:
                    list_images.append(os.path.join(root, filename))
    else:
        list_images = [x for x in os.listdir(file_dir) if os.path.splitext(x)[-1] in ext]
        list_images = [os.path.join(file_dir, x) for x in list_images]

    for file_path in  list_images:
        if num_preload <= 0:
            file_bytes =None
        else:
            with open(file_path, 'rb') as f:
                file_bytes=f.read()
        result.append((file_path, file_bytes)) 
        if len(result) == num_preload:
            break
    if len(result) == 0:
        raise RuntimeError("find no vaild files. ext = "+ext)

    return result
