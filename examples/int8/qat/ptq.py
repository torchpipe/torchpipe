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


from torchvision.models.resnet import resnet50, ResNet50_Weights
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
from torchvision import models, datasets
import torchvision
import os
import tqdm

import pytorch_quantization
from pytorch_quantization import nn as quant_nn
from pytorch_quantization import quant_modules
from pytorch_quantization import calib


tmp_dir = "./tmp"
if not os.path.exists(tmp_dir):
    os.mkdir(tmp_dir)

quant_modules.initialize()

q_model = resnet50(weights=ResNet50_Weights.DEFAULT)

model_name = q_model._get_name()
q_model.fc = nn.Linear(2048, 10)
q_model = q_model.cuda()

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
q_model.load_state_dict(modified_state_dict)


# Define main data directory
DATA_DIR = './data/imagenette2-320'
# Define training and validation data paths
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
VAL_DIR = os.path.join(DATA_DIR, 'val')


# Performing Transformations on the dataset and defining training and validation dataloaders
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])
train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=transform)
val_dataset = datasets.ImageFolder(VAL_DIR, transform=transform)
calib_dataset = torch.utils.data.random_split(val_dataset, [2901, 1024])[1]

train_dataloader = data.DataLoader(
    train_dataset, batch_size=64, shuffle=True, drop_last=True)
val_dataloader = data.DataLoader(
    val_dataset, batch_size=64, shuffle=False, drop_last=True)
# calib_dataloader = data.DataLoader(calib_dataset, batch_size=64, shuffle=False, drop_last=True)


# Declare Learning rate
lr = 0.0001

# Use cross entropy loss for classification and SGD optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(q_model.parameters(), lr=lr)


optimizer.load_state_dict(ckpt["opt_state_dict"])


def compute_amax(model, **kwargs):
    # Load calib result
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                if isinstance(module._calibrator, calib.MaxCalibrator):
                    module.load_calib_amax()
                else:
                    module.load_calib_amax(**kwargs)
    model.cuda()


def collect_stats(model, data_loader, num_batches):
    """Feed data to the network and collect statistics"""
    # Enable calibrators
    model.eval()
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                module.disable_quant()
                module.enable_calib()
            else:
                module.disable()

    # Feed data to the network for collecting stats
    for i, (image, _) in enumerate(data_loader):
        model(image.cuda())
        if i >= num_batches:
            break

    # Disable calibrators
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                module.enable_quant()
                module.disable_calib()
            else:
                module.enable()


# Calibrate the model using max calibration technique.
with torch.no_grad():
    collect_stats(q_model, train_dataloader, num_batches=64)
    compute_amax(q_model, method="max")


# Save the PTQ model
torch.save(q_model.state_dict(), f"{tmp_dir}/{model_name}_ptq.pth")


def evaluate(model, dataloader, crit, epoch):
    total = 0
    correct = 0
    loss = 0.0
    class_probs = []
    class_preds = []
    model.eval()
    with torch.no_grad():
        for data, labels in dataloader:
            data, labels = data.cuda(), labels.cuda(non_blocking=True)
            out = model(data)
            loss += crit(out, labels)
            preds = torch.max(out, 1)[1]
            class_preds.append(preds)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
    return correct / total


# Evaluate the PTQ Model
test_acc = evaluate(q_model, val_dataloader, criterion, 0)
print("{} PTQ accuracy: {:.2f}%".format(model_name, 100 * test_acc))
