#!/usr/bin/env python
"""
Debug script for PyTensorrtTensor.
This script runs each step independently to identify where it gets stuck.
"""

from __future__ import annotations

import sys
import os
import tempfile
import traceback

print("=" * 80)
print("Step 1: Check imports")
print("=" * 80)

try:
    import torch
    print(f"✓ PyTorch version: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
except Exception as e:
    print(f"✗ PyTorch import failed: {e}")
    sys.exit(1)

try:
    import tensorrt as trt
    print(f"✓ TensorRT version: {trt.__version__}")
except Exception as e:
    print(f"✗ TensorRT import failed: {e}")
    traceback.print_exc()
    sys.exit(1)

try:
    from torchpipe.backends.py_tensorrt import PyTensorrtEngine, PyTensorrtInferTensor, ProfileConfig
    from torchpipe.backends.trt_utils import onnx_to_trt, get_engine_io_info, create_context
    print("✓ TorchPipe backends imported")
except Exception as e:
    print(f"✗ TorchPipe backends import failed: {e}")
    traceback.print_exc()
    sys.exit(1)

print()
print("=" * 80)
print("Step 2: Create test ONNX model")
print("=" * 80)

tmp_onnx = None
try:
    class Identity(torch.nn.Module):
        def forward(self, x):
            return x * 2
    
    model = Identity()
    model.eval()
    
    tmp_onnx = tempfile.mktemp(suffix='.onnx')
    dummy_input = torch.randn(1, 3, 224, 224)
    
    torch.onnx.export(
        model,
        dummy_input,
        tmp_onnx,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print(f"✓ ONNX model created: {tmp_onnx}")
    print(f"  File size: {os.path.getsize(tmp_onnx)} bytes")
except Exception as e:
    print(f"✗ ONNX model creation failed: {e}")
    traceback.print_exc()
    if tmp_onnx and os.path.exists(tmp_onnx):
        os.remove(tmp_onnx)
    sys.exit(1)

print()
print("=" * 80)
print("Step 3: Build TRT engine directly with onnx_to_trt")
print("=" * 80)

engine = None
try:
    from torchpipe.backends.trt_utils import onnx_to_trt
    print("Calling onnx_to_trt...")
    
    profiles = [
        ProfileConfig(
            min_shapes={'input': (1, 3, 224, 224)},
            opt_shapes={'input': (4, 3, 224, 224)},
            max_shapes={'input': (8, 3, 224, 224)},
        )
    ]
    
    engine = onnx_to_trt(
        onnx_path=tmp_onnx,
        max_batch_size=8,
        fp16_mode=False,
        profiles=profiles
    )
    print("✓ Engine built successfully!")
    print(f"  num_bindings: {engine.num_bindings}")
    print(f"  num_optimization_profiles: {engine.num_optimization_profiles}")
except Exception as e:
    print(f"✗ Engine building failed: {e}")
    traceback.print_exc()
    if tmp_onnx and os.path.exists(tmp_onnx):
        os.remove(tmp_onnx)
    sys.exit(1)

print()
print("=" * 80)
print("Step 4: Get IO info")
print("=" * 80)

try:
    io_info = get_engine_io_info(engine)
    print("✓ IO info retrieved")
    print(f"  Inputs: {len(io_info[0])}")
    print(f"  Outputs: {len(io_info[1])}")
    
    for i, inp in enumerate(io_info[0]):
        print(f"    Input {i}: {inp.name}, min={inp.min.to_tuple()}, max={inp.max.to_tuple()}")
    
    for i, out in enumerate(io_info[1]):
        print(f"    Output {i}: {out.name}, min={out.min.to_tuple()}, max={out.max.to_tuple()}")
except Exception as e:
    print(f"✗ IO info retrieval failed: {e}")
    traceback.print_exc()
    if tmp_onnx and os.path.exists(tmp_onnx):
        os.remove(tmp_onnx)
    sys.exit(1)

print()
print("=" * 80)
print("Step 5: Create execution context")
print("=" * 80)

try:
    context = create_context(engine, 0)
    print("✓ Execution context created!")
    print(f"  Context type: {type(context)}")
except Exception as e:
    print(f"✗ Context creation failed: {e}")
    traceback.print_exc()
    if tmp_onnx and os.path.exists(tmp_onnx):
        os.remove(tmp_onnx)
    sys.exit(1)

print()
print("=" * 80)
print("Step 6: Test PyTensorrtEngine")
print("=" * 80)

try:
    py_engine = PyTensorrtEngine(instance_num=1)
    print("✓ PyTensorrtEngine created")
    
    profiles = [
        ProfileConfig(
            min_shapes={'input': (1, 3, 224, 224)},
            opt_shapes={'input': (4, 3, 224, 224)},
            max_shapes={'input': (8, 3, 224, 224)},
        )
    ]
    py_engine.load_from_onnx(tmp_onnx, profiles=profiles)
    print("✓ PyTensorrtEngine.load_from_onnx completed")
    
    io_info2 = py_engine.get_io_info()
    print(f"  IO info from engine: {len(io_info2[0])} inputs, {len(io_info2[1])} outputs")
    
    context2 = py_engine.create_context(0)
    print("✓ PyTensorrtEngine.create_context completed")
except Exception as e:
    print(f"✗ PyTensorrtEngine test failed: {e}")
    traceback.print_exc()
    if tmp_onnx and os.path.exists(tmp_onnx):
        os.remove(tmp_onnx)
    sys.exit(1)

print()
print("=" * 80)
print("Step 7: Test PyTensorrtInferTensor")
print("=" * 80)

try:
    backend = PyTensorrtInferTensor()
    print("✓ PyTensorrtInferTensor created")
    
    config = {
        "model": tmp_onnx,
        "model_type": "onnx",
        "instance_num": "1",
        "instance_index": "0",
    }
    
    print("Calling PyTensorrtInferTensor.init...")
    backend.init(config)
    print("✓ PyTensorrtInferTensor.init completed!")
except Exception as e:
    print(f"✗ PyTensorrtInferTensor test failed: {e}")
    traceback.print_exc()
    if tmp_onnx and os.path.exists(tmp_onnx):
        os.remove(tmp_onnx)
    sys.exit(1)

print()
print("=" * 80)
print("✓ All steps passed!")
print("=" * 80)

if tmp_onnx and os.path.exists(tmp_onnx):
    os.remove(tmp_onnx)
    print(f"  Cleaned up: {tmp_onnx}")
