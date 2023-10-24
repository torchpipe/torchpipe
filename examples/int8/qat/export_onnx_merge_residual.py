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

from pytorch_quantization import calib
from pytorch_quantization import quant_modules
from pytorch_quantization import nn as quant_nn
import pytorch_quantization
import calib_tools
from torchvision.models.resnet import resnet50, ResNet50_Weights
import os
import torch
from torch import nn

tmp_dir = "./tmp"
if not os.path.exists(tmp_dir):
    os.mkdir(tmp_dir)


fp32_model = resnet50(weights=None)

model_name = fp32_model._get_name()
fp32_model.fc = nn.Linear(2048, 10)
fp32_model = fp32_model.cuda()

# mobilenetv2_base_ckpt is the checkpoint generated from Step 2 : Training a baseline Mobilenetv2 model.
ckpt = torch.load(os.path.join(tmp_dir, f"{model_name}_base_ckpt"))
modified_state_dict = {}
for key, val in ckpt["model_state_dict"].items():
    # Remove 'module.' from the key names
    if key.startswith('module'):
        modified_state_dict[key[7:]] = val
    else:
        modified_state_dict[key] = val

# Load the pre-trained checkpoint
fp32_model.load_state_dict(modified_state_dict)


calib_tools.save_onnx(fp32_model, f"{tmp_dir}/{model_name}_fp32.onnx")


calib = calib_tools.Calibrator("mse")

if False:
    from torchvision.models.resnet import resnet50, ResNet50_Weights
else:
    from quant_models.resnet import resnet50, ResNet50_Weights

q_model = resnet50(weights=ResNet50_Weights.DEFAULT)


model_name = q_model._get_name()
q_model.fc = nn.Linear(2048, 10)
q_model = q_model.cuda()

# mobilenetv2_base_ckpt is the checkpoint generated from Step 2 : Training a baseline Mobilenetv2 model.
ckpt = torch.load(os.path.join(tmp_dir, f"{model_name}_ptq.pth"))

modified_state_dict = {}
for key, val in ckpt.items():
    # Remove 'module.' from the key names
    if key.startswith('module'):
        modified_state_dict[key[7:]] = val
    else:
        modified_state_dict[key] = val

# Load the pre-trained checkpoint
q_model.load_state_dict(modified_state_dict)


calib_tools.save_onnx(q_model, f"{tmp_dir}/{model_name}_ptq.onnx")


# mobilenetv2_base_ckpt is the checkpoint generated from Step 2 : Training a baseline Mobilenetv2 model.
fp32_model_path = os.path.join(tmp_dir, f"{model_name}_qat_ckpt")
ckpt = torch.load(fp32_model_path)

modified_state_dict = {}
print(f"{fp32_model_path} loaded, acc=", ckpt["acc"])
for key, val in ckpt["model_state_dict"].items():
    # Remove 'module.' from the key names
    if key.startswith('module'):
        modified_state_dict[key[7:]] = val
    else:
        modified_state_dict[key] = val

# Load the pre-trained checkpoint
q_model.load_state_dict(modified_state_dict)
calib_tools.save_onnx(q_model, f"{tmp_dir}/{model_name}_qat.onnx")

cmd = f"/opt/tensorrt/bin/trtexec --onnx={tmp_dir}/{model_name}_fp32.onnx --best --saveEngine={tmp_dir}/{model_name}_fp32.trt"
print(cmd)
cmd = f"/opt/tensorrt/bin/trtexec --onnx={tmp_dir}/{model_name}_ptq.onnx --best --saveEngine={tmp_dir}/{model_name}_ptq.trt"
print(cmd)
cmd = f"/opt/tensorrt/bin/trtexec --onnx={tmp_dir}/{model_name}_qat.onnx --best --saveEngine={tmp_dir}/{model_name}_qat.trt"
print(cmd)

# /opt/tensorrt/bin/trtexec --onnx=./tmp/ResNet_fp32.onnx --best --saveEngine=./tmp/ResNet_fp32.trt
# /opt/tensorrt/bin/trtexec --onnx=./tmp/ResNet_ptq.onnx --best --saveEngine=./tmp/ResNet_ptq.trt
# /opt/tensorrt/bin/trtexec --onnx=./tmp/ResNet_qat.onnx --best --saveEngine=./tmp/ResNet_qat.trt
