#!/usr/bin/env python
"""
Advanced example: Multi-profile inference with dynamic shapes.

This example demonstrates how to use multiple optimization profiles
for efficient inference with varying input shapes.
"""

import torch
import tempfile
import os

from torchpipe.backends.py_tensorrt import PyTensorrtEngine, PyTensorrtInferTensor, ProfileConfig

# Step 1: Create a Conv model
class ConvModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = torch.nn.MaxPool2d(2, 2)
    
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        return x

model = ConvModel()
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

# Step 3: Create engine with multiple profiles
engine = PyTensorrtEngine(instance_num=2)

profiles = [
    ProfileConfig(
        min_shapes={'input': (1, 3, 224, 224)},
        opt_shapes={'input': (4, 3, 224, 224)},
        max_shapes={'input': (8, 3, 224, 224)},
    ),
    ProfileConfig(
        min_shapes={'input': (1, 3, 224, 224)},
        opt_shapes={'input': (2, 3, 224, 224)},
        max_shapes={'input': (4, 3, 224, 224)},
    )
]

engine.load_from_onnx(
    tmp_onnx,
    profiles=profiles,
    fp16_mode=True
)

print(f"Engine created with {engine.num_profiles} profiles")

# Step 4: Create backend using the engine
backend = PyTensorrtInferTensor()
backend._engine = engine
backend._context = engine.get_or_create_context(0)
backend._io_info = engine.get_io_info(0)
backend._initialized = True
backend._input_finish_event = torch.cuda.Event()

# Step 5: Run inference with different batch sizes
print("\nTesting with different batch sizes:")
for batch_size in [1, 2, 4, 8]:
    input_tensor = torch.randn((batch_size, 3, 224, 224), device='cuda')
    io_dict = {"data": input_tensor}
    
    backend.forward([io_dict])
    
    result = io_dict["result"]
    print(f"  Batch {batch_size}: input {input_tensor.shape} -> output {result.shape}")
    
    # Verify shape
    assert result.shape[0] == batch_size

print("\n✓ Multi-profile inference successful!")

# Cleanup
os.remove(tmp_onnx)
