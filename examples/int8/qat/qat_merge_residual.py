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


from quant_models.resnet import resnet50, ResNet50_Weights
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


if False:
    quant_modules.initialize()
else:
    import calib_tools
    calib = calib_tools.Calibrator("mse")


# q_model = resnet50(weights=ResNet50_Weights.DEFAULT)
q_model = resnet50()

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


# Define functions for training, evalution, saving checkpoint and train parameter setting function
def train(model, dataloader, crit, opt, epoch):
    model.train()
    running_loss = 0.0
    for batch, (data, labels) in enumerate(dataloader):
        data, labels = data.cuda(), labels.cuda(non_blocking=True)
        opt.zero_grad()
        out = model(data)
        loss = crit(out, labels)
        loss.backward()
        opt.step()
        running_loss += loss.item()
        if batch % 100 == 99:
            print("Batch: [%5d | %5d] loss: %.3f" %
                  (batch + 1, len(dataloader), running_loss / 100))
            running_loss = 0.0


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


# Declare Learning rate
lr = 0.0001

# Use cross entropy loss for classification and SGD optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(q_model.parameters(), lr=lr)


optimizer = optim.SGD(q_model.parameters(), lr=lr)

# Evaluate the PTQ Model
test_acc = evaluate(q_model, val_dataloader, criterion, 0)
print("{} PTQ accuracy: {:.2f}%".format(model_name, 100 * test_acc))


# Finetune the QAT model for 2 epochs
num_epochs = 2

for epoch in range(num_epochs):
    print('Epoch: [%5d / %5d] LR: %f' % (epoch + 1, num_epochs, lr))

    train(q_model, train_dataloader, criterion, optimizer, epoch)
    test_acc = evaluate(q_model, val_dataloader, criterion, epoch)

    print("Test Acc: {:.2f}%".format(100 * test_acc))


def save_checkpoint(state, ckpt_path="checkpoint.pth"):
    torch.save(state, ckpt_path)
    print("Checkpoint saved")


save_checkpoint({'epoch': epoch + 1,
                 'model_state_dict': q_model.state_dict(),
                 'acc': test_acc,
                 'opt_state_dict': optimizer.state_dict()
                 },
                ckpt_path=f"{tmp_dir}/{model_name}_qat_ckpt")
