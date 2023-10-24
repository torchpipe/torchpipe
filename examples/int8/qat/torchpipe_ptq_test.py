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
train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=transform)

train_dataloader = data.DataLoader(
    train_dataset, batch_size=1, shuffle=True, drop_last=True)
val_dataloader = data.DataLoader(
    val_dataset, batch_size=64, shuffle=False, drop_last=True)


def has_cache_data(cache_data_dir):
    print(f"cache_data_dir: {cache_data_dir}")
    assert (os.path.exists(cache_data_dir))
    files = [x for x in os.listdir(cache_data_dir) if x.endswith(".pt")]
    if len(files) == 0:
        return False
    else:
        return True
        print(f"found {len(files)} data in {cache_data_dir}")


def load_cache_onnx(onnx_path, cache_data_dir):
    assert (os.path.exists(onnx_path))

    class TorchpipeModel:
        def __init__(self, tp_model):
            self.tp = tp_model

        def __call__(self, data):
            input = {"data": data}

            self.tp(input)
            return input["result"]

    print(f"cache_data_dir: {cache_data_dir}")

    import torchpipe as tp
    config = {"backend": "SyncTensor[TensorrtTensor]",
              "model": onnx_path,
              "precision": "fp32",
              # "model::cache":path.replace(".onnx",".trt"),
              "max": 64,
              "instance_num": 1,
              "batching_timeout": 0}

    config["backend"] = "SaveTensor"
    config["save_dir"] = f"{cache_data_dir}"

    # config["precision::fp16"]=f"fc.weight,/fc/Gemm,fc.bias"
    # config["precision::fp16"]=f"proj_head.0.weight,/proj_head/proj_head.0/Gemm,proj_head.0.bias"

    # config["model::cache"]=path.replace(".onnx","")+f"_{precision}.trt"
    # config["precision::fp16"]=f"fc.weight,/fc/Gemm,fc.bias"
    # config["precision::fp16"]=f"proj_head.0.weight,/proj_head/proj_head.0/Gemm,proj_head.0.bias"
    model = tp.pipe(config)
    return TorchpipeModel(model)


def load_ptq_onnx(onnx_path, cache_data_dir):
    assert (os.path.exists(onnx_path))

    class TorchpipeModel:
        def __init__(self, tp_model):
            self.tp = tp_model

        def __call__(self, data):
            input = {"data": data}

            self.tp(input)
            return input["result"]

    print(f"cache_data_dir: {cache_data_dir}")
    assert (os.path.exists(cache_data_dir))
    files = [x for x in os.listdir(cache_data_dir) if x.endswith(".pt")]

    precision = "best"
    print(f"found {len(files)} data in {cache_data_dir}")
    assert (len(files) > 10)

    import torchpipe as tp
    config = {"backend": "SyncTensor[TensorrtTensor]",
              "model": onnx_path,
              "precision": precision,
              # "model::cache":path.replace(".onnx",".trt"),
              "max": 64,
              "instance_num": 1,
              "batching_timeout": 0}

    config["calibrate_input"] = cache_data_dir
    # config["model::cache"]= onnx_path.replace(".onnx","")+f"_{precision}_ptq.trt"
    # config["precision::fp16"]=f"fc.weight,/fc/Gemm,fc.bias"
    # config["precision::fp16"]=f"proj_head.0.weight,/proj_head/proj_head.0/Gemm,proj_head.0.bias"

    # config["model::cache"]=path.replace(".onnx","")+f"_{precision}.trt"
    # config["precision::fp16"]=f"fc.weight,/fc/Gemm,fc.bias"
    # config["precision::fp16"]=f"proj_head.0.weight,/proj_head/proj_head.0/Gemm,proj_head.0.bias"
    model = tp.pipe(config)
    return TorchpipeModel(model)


def evaluate(model, dataloader, crit, epoch):
    total = 0
    correct = 0
    loss = 0.0
    class_probs = []
    class_preds = []

    for data, labels in dataloader:
        data, labels = data.cuda(), labels.cuda(non_blocking=True)
        out = model(data)
        loss += crit(out, labels)
        preds = torch.max(out, 1)[1]
        class_preds.append(preds)
        total += labels.size(0)
        correct += (preds == labels).sum().item()
    return correct / total


criterion = nn.CrossEntropyLoss()

tmp_dir = "./tmp"
cache_dir = "./tmp/cache_data/"
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)
model_name = "ResNet"
if not has_cache_data(cache_dir):
    model = load_cache_onnx(f"{tmp_dir}/{model_name}_fp32.onnx", cache_dir)
    num_batches = 500
    for data, labels in train_dataloader:
        assert data.shape[0] == 1
        data, labels = data.cuda(), labels.cuda(non_blocking=True)
        out = model(data)
        num_batches -= 1
        if num_batches < 0:
            break

q_model = load_ptq_onnx(f"{tmp_dir}/{model_name}_fp32.onnx", cache_dir)
test_acc = evaluate(q_model, val_dataloader, criterion, 0)
print("{}  accuracy: {:.2f}%".format("tp", 100 * test_acc))
