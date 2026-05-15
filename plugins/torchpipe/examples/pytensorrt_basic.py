#!/usr/bin/env python
"""
Basic example: Using PyTensorrtInferTensor for inference.

This example demonstrates the basic usage of PyTensorrtInferTensor
for running inference on a simple model.
"""

import torch
import tempfile
import os

from torchpipe.backends.py_tensorrt import PyTensorrtInferTensor

# Step 1: Create a simple PyTorch model
class SimpleModel(torch.nn.Module):
    def forward(self, x):
        return x * 2

model = SimpleModel()
model.eval()

# Step 2: Export to ONNX
tmp_onnx = tempfile.NamedTemporaryFile(suffix='.onnx', delete=False).name
dummy_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(
    model,
    dummy_input,
    tmp_onnx,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
)

print(f"ONNX model created: {tmp_onnx}")

# Step 3: Initialize PyTensorrtInferTensor backend
backend = PyTensorrtInferTensor()

config = {
    "model": tmp_onnx,
    "model_type": "onnx",
    "instance_num": "1",
}

backend.init(config)
print("Backend initialized successfully")

# Step 4: Run inference
input_tensor = torch.ones((4, 3, 224, 224), device='cuda')
io_dict = {"data": input_tensor}

backend.forward([io_dict])

result = io_dict["result"]
print(f"Input shape: {input_tensor.shape}")
print(f"Output shape: {result.shape}")
print(f"Output dtype: {result.dtype}")

# Step 5: Verify result
expected = input_tensor * 2
if torch.allclose(result, expected, rtol=1e-2, atol=1e-2):
    print("✓ Inference result is correct!")
else:
    print("✗ Inference result mismatch!")

# Cleanup
os.remove(tmp_onnx)
