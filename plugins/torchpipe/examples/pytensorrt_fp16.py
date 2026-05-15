#!/usr/bin/env python
"""
FP16 example: High-performance inference with FP16 precision.

This example demonstrates how to use FP16 mode for faster inference
on GPUs with Tensor Cores.
"""

import torch
import tempfile
import os
import time

from torchpipe.backends.py_tensorrt import PyTensorrtInferTensor

# Step 1: Create a ResNet-like model
class ResNetBlock(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(channels)
        self.conv2 = torch.nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(channels)
    
    def forward(self, x):
        residual = x
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return torch.relu(out)

class SimpleResNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = torch.nn.BatchNorm2d(64)
        self.pool = torch.nn.MaxPool2d(3, stride=2, padding=1)
        self.layer1 = ResNetBlock(64)
        self.layer2 = ResNetBlock(64)
        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.fc = torch.nn.Linear(64, 10)
    
    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

model = SimpleResNet()
model.eval()
model = model.cuda().half()

# Step 2: Export to ONNX with FP16
tmp_onnx = tempfile.mktemp(suffix='.onnx')
dummy_input = torch.randn(1, 3, 224, 224, dtype=torch.float16, device='cuda')
torch.onnx.export(
    model,
    dummy_input,
    tmp_onnx,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
)

print(f"ONNX model created: {tmp_onnx}")

# Step 3: Initialize backend with FP16 mode
backend = PyTensorrtInferTensor()

config = {
    "model": tmp_onnx,
    "model_type": "onnx",
    "instance_num": "1",
}

backend.init(config)
print("Backend initialized with FP16 mode")

# Step 4: Benchmark inference
print("\nBenchmarking FP16 inference:")
batch_sizes = [1, 4, 8]

for batch_size in batch_sizes:
    input_tensor = torch.randn((batch_size, 3, 224, 224), dtype=torch.float16, device='cuda')
    
    # Warmup
    for _ in range(10):
        io_dict = {"data": input_tensor}
        backend.forward([io_dict])
    
    torch.cuda.synchronize()
    
    # Benchmark
    num_iterations = 100
    start_time = time.time()
    
    for _ in range(num_iterations):
        io_dict = {"data": input_tensor}
        backend.forward([io_dict])
    
    torch.cuda.synchronize()
    elapsed_time = time.time() - start_time
    
    latency = elapsed_time / num_iterations * 1000
    throughput = batch_size / (elapsed_time / num_iterations)
    
    print(f"  Batch {batch_size:2d}: latency = {latency:.2f} ms, throughput = {throughput:.1f} img/s")

print("\n✓ FP16 inference benchmark complete!")

# Cleanup
os.remove(tmp_onnx)
