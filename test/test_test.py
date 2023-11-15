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

input = list(range(0, 10000))
result = {}


class FileSampler(torchpipe.utils.test.SequentialSampler):
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
        return


def test_test_random(dir="assets/norm_jpg/"):
    from typing import List, Tuple
    import torchpipe as tp

    model = tp.pipe({"backend": "DecodeMat", "instance_num": 10})

    def run(imgs: List[Tuple[str, bytes]]):
        img_path, img_bytes = imgs[0]
        input = {"data": img_bytes}
        model(input)

        assert (input["result"].shape == (576, 768, 3))

    torchpipe.utils.test.test_from_raw_file(run, os.path.join(dir),
                                            num_clients=10, batch_size=1, total_number=10000)


def test_test_from_raw_file(dir="assets/norm_jpg/"):
    from typing import List, Tuple
    import torchpipe as tp

    model = tp.pipe({"backend": "DecodeMat", "instance_num": 10})

    def run(imgs: List[Tuple[str, bytes]]):
        img_path, img_bytes = imgs[0]
        input = {"data": img_bytes}
        model(input)

        assert (input["result"].shape == (576, 768, 3))

    torchpipe.utils.test.test_from_raw_file(run, os.path.join(dir),
                                            num_clients=10, batch_size=1, num_preload=0)


def test_test_function(dir="assets/norm_jpg/"):
    from typing import List, Tuple
    import torchpipe as tp

    model = tp.pipe({"backend": "DecodeMat", "instance_num": 10})
    file_path = "assets/norm_jpg/dog.jpg"
    with open(file_path, 'rb') as f:
        file_bytes = f.read()

    def run():
        input = {"data": file_bytes}
        model(input)

        assert (input["result"].shape == (576, 768, 3))
    num_clients = 5
    torchpipe.utils.test.test_function(
        [run]*num_clients, num_clients=num_clients, batch_size=1, total_number=1000)


def test_all_files(file_dir: str = "assets/norm_jpg/", num_clients=10, batch_size=1,
                   ext=[".jpg", '.JPG', '.jpeg', '.JPEG']):

    files = [x for x in os.listdir(file_dir) if os.path.splitext(x)[-1] in ext]
    files = [os.path.join(file_dir, x) for x in files]

    forwards = [FileSampler(files, batch_size) for i in range(num_clients)]

    torchpipe.utils.test.test(forwards, len(files))


if __name__ == "__main__":
    # test_test_random(dir = "assets/norm_jpg/")

    # test_all_files(file_dir = "../examples/ocr_poly_v2/test_img/", num_clients=10)
    test_test_from_raw_file()
