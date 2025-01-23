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
# from torchvision import models
import os
import tempfile


class TestUtilsImport:
    @classmethod
    def setup_class(self):

        pass

    def test_torchpipe_module_import(self):
        import torchpipe
        torchpipe.utils.models.register_model
        torchpipe.utils.models.create_model
        torchpipe.utils.models.list_models
        torchpipe.utils.models.register_model_from_timm
        torchpipe.utils.models.onnx_export

        torchpipe.utils.test.FileSampler
        torchpipe.utils.test.Sampler
        torchpipe.utils.test.RandomSampler
        torchpipe.utils.test.SequentialSampler
        torchpipe.utils.test.preload
        torchpipe.utils.test.test
        torchpipe.utils.test.test_function
        torchpipe.utils.test.test_functions
        torchpipe.utils.test.test_from_raw_file
        torchpipe.utils.test.throughput

    def test_torchpipe_model_import_from(self):

        from torchpipe.utils.models import register_model
        from torchpipe.utils.models import create_model
        from torchpipe.utils.models import list_models
        from torchpipe.utils.models import register_model_from_timm
        from torchpipe.utils.models import onnx_export

        from torchpipe.utils.test import FileSampler
        from torchpipe.utils.test import Sampler
        from torchpipe.utils.test import RandomSampler
        from torchpipe.utils.test import SequentialSampler
        from torchpipe.utils.test import preload
        from torchpipe.utils.test import test
        from torchpipe.utils.test import test_function
        from torchpipe.utils.test import test_functions
        from torchpipe.utils.test import test_from_raw_file
        from torchpipe.utils.test import throughput

    # def test_torchpipe_utils_tools(self):
    #     import os
    #     os.system("python -m torchpipe.utils.encrypt ./assets/Identity_4.onnx ./assets/Identity_4.onnx.encrypted")
    #     os.system("python -m torchpipe.utils.vis ./assets/PipelineV3.toml")


if __name__ == "__main__":

    a = TestUtilsImport()
    a.setup_class()
    a.test_torchpipe_module_import()
    a.test_torchpipe_model_import_from()
    # a.test_torchpipe_utils_tools()
