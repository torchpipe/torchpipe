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

from torchvision.models.resnet import resnet50, ResNet50_Weights
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
from torchvision import models, datasets


import os
import tensorrt as trt
import numpy as np
import time
import wget
import tarfile
import shutil

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


model = resnet50(weights=ResNet50_Weights.DEFAULT)

model_name = model._get_name()


model.fc = nn.Linear(2048, 10)
model = model.cuda()


# Declare Learning rate
lr = 0.0001

# Use cross entropy loss for classification and SGD optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=lr)


tmp_dir = "./tmp"
if not os.path.exists(tmp_dir):
    os.mkdir(tmp_dir)

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


def save_checkpoint(state, ckpt_path="checkpoint.pth"):
    torch.save(state, ckpt_path)
    print("Checkpoint saved")

# Helper function to benchmark the model
# torch.backends.cudnn.benchmark = True


# Train the model for 20 epochs to attain an acceptable accuracy.
num_epochs = 40
for epoch in range(num_epochs):
    print('Epoch: [%5d / %5d] LR: %f' % (epoch + 1, num_epochs, lr))

    train(model, train_dataloader, criterion, optimizer, epoch)
    test_acc = evaluate(model, val_dataloader, criterion, epoch)

    print("Test Acc: {:.2f}%".format(100 * test_acc))

save_checkpoint({'epoch': epoch + 1,
                 'model_state_dict': model.state_dict(),
                 'acc': test_acc,
                 'opt_state_dict': optimizer.state_dict()
                 },
                ckpt_path=f"{tmp_dir}/{model_name}_base_ckpt")


# Evaluate the baseline model
test_acc = evaluate(model, val_dataloader, criterion, 0)
print("{} Baseline accuracy: {:.2f}%".format(model_name, 100 * test_acc))
