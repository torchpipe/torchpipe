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


import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
from torchvision import models, datasets
import torchvision
import os
import tqdm


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
val_dataset = datasets.ImageFolder(VAL_DIR, transform=transform)
# train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=transform)

# train_dataloader = data.DataLoader(train_dataset, batch_size=1, shuffle=True, drop_last=True)
val_dataloader = data.DataLoader(
    val_dataset, batch_size=64, shuffle=False, drop_last=True)


class OnnxRuntimeModel:
    def __init__(self, sess):
        self.sess = sess

        import onnxruntime as rt
        self.device = rt.get_device()

    def __call__(self, data):
        input = {"input": data.numpy()}

        pred_onx = self.sess.run(None, input)
        # print(pred_onx)
        return torch.from_numpy(pred_onx[0]).cuda()


def load_onnx(path):
    import onnxruntime as rt
    device = rt.get_device()
    assert (device == "GPU")
    print("you need `pip install onnxruntime-gpu`")
    if device == "GPU":  # CUDAExecutionProvider TensorrtExecutionProvider
        # return OnnxRuntimeModel(rt.InferenceSession(path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider']))
        return OnnxRuntimeModel(rt.InferenceSession(path, providers=['CUDAExecutionProvider']))
    else:
        return OnnxRuntimeModel(rt.InferenceSession(path))


def evaluate(model, dataloader, crit, epoch):
    total = 0
    correct = 0
    loss = 0.0
    class_probs = []
    class_preds = []

    for data, labels in dataloader:
        # data, labels = data.cuda(), labels.cuda(non_blocking=True)
        labels = labels.cuda()
        out = model(data).cuda()
        loss += crit(out, labels)
        preds = torch.max(out, 1)[1]
        class_preds.append(preds)
        total += labels.size(0)
        correct += (preds == labels).sum().item()
    return correct / total


criterion = nn.CrossEntropyLoss()

tmp_dir = "./tmp"

model_name = "ResNet"

q_model = load_onnx(f"{tmp_dir}/{model_name}_qat.onnx")
test_acc = evaluate(q_model, val_dataloader, criterion, 0)
print("{}  accuracy: {:.2f}%".format("tp", 100 * test_acc))
