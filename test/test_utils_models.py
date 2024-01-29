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

import torchpipe
import torch
from torchvision import models
import torchvision
import os
import tempfile


class TestUtilsModels:
    @classmethod
    def setup_class(self):

        tmpdir = tempfile.gettempdir()
        self.onnx_path = os.path.join(tmpdir, "resnet18.onnx")
        if torchvision.__version__ >= "0.9.0":
            self.model = models.resnet18(weights=None).eval()
        else:
            self.model = models.resnet18(pretrained=False).eval()
        self.input = torch.randn(1, 3, 224, 224)
        self.dict_args = {
            "model": self.onnx_path,
            "precision": "fp16",
        }

    def test_register_model(self):
        from torchpipe.utils.models import register_model, create_model, list_models, register_model_from_timm
        import shutil

        @register_model
        def resnet18(**kwargs):
            return models.resnet18(**kwargs)

        register_model_from_timm(model_name="resnet10t")

        all_models = list_models()

        assert "resnet18" in all_models
        assert "resnet10t" in all_models

        for model_name in all_models:
            print(f"test model {model_name}")
            model = create_model(model_name, num_classes=3).eval()

            input = torch.randn(1, 3, 224, 224)

            output = model(input)

    # There is a dependency

    def test_onnx_export_throughput(self):
        torchpipe.utils.models.onnx_export(
            self.model, self.onnx_path, self.input)
        result = torchpipe.utils.test.throughput(
            self.dict_args, num_clients=5, total_number=100)
        assert isinstance(result, dict)


if __name__ == "__main__":

    a = TestUtilsModels()
    a.setup_class()
    a.test_onnx_export_throughput()
    a.test_register_model()
