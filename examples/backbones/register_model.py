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


import timm
import torch
import cv2
import numpy as np
import os
import torchpipe as tp
from torchpipe import pipe, TASK_DATA_KEY, TASK_RESULT_KEY, TASK_BOX_KEY
from torchpipe.utils.models import onnx_export

from torchpipe.utils.models import register_model, create_model, list_models,  register_model_from_timm
import shutil


@register_model
def resnet50(pretrained=False, **kwargs):
    return timm.create_model("resnet50", pretrained=pretrained, **kwargs)


@register_model
def resnet18(pretrained=False, **kwargs):
    return timm.create_model("resnet50", pretrained=pretrained, **kwargs)

register_model_from_timm(model_name="resnet34")

if __name__ == "__main__":
    
    all_models = list_models()

    print(all_models)

    for model_name in all_models:
        print(f"test model {model_name}")
        model = create_model(model_name, pretrained=False, num_classes=3).eval()

        input = torch.ones(1, 3, 224, 224)

        output = model(input)

        print(output)

