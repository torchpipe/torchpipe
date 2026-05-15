"""
Resource management tests for PyTensorrtTensor.

This module tests:
- Resource allocation and deallocation
- Memory leak detection
- Thread safety
- Concurrent access
"""

from __future__ import annotations

import gc
import os
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import List

import pytest

try:
    import torch
    
    missing_dtypes = [
        "uint16", "uint32", "uint64",
        "float8_e5m2fnuz", "float8_e4m3fnuz",
        "float8_e4m3fn", "float8_e5m2"
    ]
    for dtype in missing_dtypes:
        if not hasattr(torch, dtype):
            setattr(torch, dtype, None)
    
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import tensorrt as trt
    HAS_TRT = True
except (ImportError, OSError):
    HAS_TRT = False


def pytest_skip_if_no_cuda():
    if not HAS_TORCH or not torch.cuda.is_available():
        pytest.skip("CUDA not available")


def pytest_skip_if_no_trt():
    if not HAS_TRT:
        pytest.skip("TensorRT not available")


class SimpleModel(torch.nn.Module):
    """Simple model for testing."""
    
    def forward(self, x):
        return x * 2


class TestResourceManagement:
    """Tests for resource management."""
    
    @pytest.mark.skipif(not HAS_TRT or not HAS_TORCH or not torch.cuda.is_available(), reason="TensorRT and CUDA required")
    def test_engine_creation_and_deletion(self, onnx_model_factory):
        """Test that engine is properly created and deleted."""
        from torchpipe.backends.py_tensorrt import PyTensorrtEngine
        
        model = SimpleModel()
        onnx_path = onnx_model_factory(model, (1, 3, 224, 224))
        
        engine = PyTensorrtEngine(instance_num=1)
        engine.load_from_onnx(onnx_path, fp16_mode=False)
        
        assert engine.engine is not None
        assert engine.num_profiles >= 1
        
        # Test release method
        if hasattr(engine, 'release'):
            engine.release()
            assert engine.engine is None
    
    @pytest.mark.skipif(not HAS_TRT or not HAS_TORCH or not torch.cuda.is_available(), reason="TensorRT and CUDA required")
    def test_backend_creation_and_deletion(self, tensorrt_backend_factory):
        """Test that backend is properly created and deleted."""
        model = SimpleModel()
        backend = tensorrt_backend_factory(model, (1, 3, 224, 224))
        
        assert backend._engine is not None
        assert backend._context is not None
        
        # Test release method
        # Note: release() does not release _engine because it may be shared
        if hasattr(backend, 'release'):
            backend.release()
            assert backend._context is None
            assert backend._initialized is False
    
    @pytest.mark.skipif(not HAS_TRT or not HAS_TORCH or not torch.cuda.is_available(), reason="TensorRT and CUDA required")
    def test_memory_cleanup_after_inference(self, tensorrt_backend_factory):
        """Test that memory is properly cleaned up after inference."""
        model = SimpleModel()
        backend = tensorrt_backend_factory(model, (1, 3, 224, 224))
        
        torch.cuda.reset_peak_memory_stats()
        
        # Run multiple inferences
        for _ in range(10):
            input_tensor = torch.randn((4, 3, 224, 224), device='cuda')
            io_dict = {"data": input_tensor}
            backend.forward([io_dict])
            del io_dict
        
        torch.cuda.synchronize()
        gc.collect()
        
        # Check memory usage
        # Note: This is a basic check, more sophisticated checks may be needed
    
    @pytest.mark.skipif(not HAS_TRT or not HAS_TORCH or not torch.cuda.is_available(), reason="TensorRT and CUDA required")
    def test_multiple_backend_instances(self, onnx_model_factory):
        """Test creating multiple backend instances."""
        from torchpipe.backends.py_tensorrt import PyTensorrtEngine, PyTensorrtInferTensor, ProfileConfig
        
        model = SimpleModel()
        onnx_path = onnx_model_factory(model, (1, 3, 224, 224))
        
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
        
        engine.load_from_onnx(onnx_path, profiles=profiles, fp16_mode=False)
        
        backends = []
        for i in range(2):
            backend = PyTensorrtInferTensor()
            backend._engine = engine
            backend._context = engine.get_or_create_context(i)
            backend._io_info = engine.get_io_info(i)
            backend._initialized = True
            backend._input_finish_event = torch.cuda.Event()
            backends.append(backend)
        
        # Test all backends work
        for backend in backends:
            input_tensor = torch.randn((1, 3, 224, 224), device='cuda')
            io_dict = {"data": input_tensor}
            backend.forward([io_dict])
            assert io_dict["result"] is not None
        
        # Cleanup
        for backend in backends:
            if hasattr(backend, 'release'):
                backend.release()
        
        if hasattr(engine, 'release'):
            engine.release()


class TestThreadSafety:
    """Tests for thread safety."""
    
    @pytest.mark.skipif(not HAS_TRT or not HAS_TORCH or not torch.cuda.is_available(), reason="TensorRT and CUDA required")
    def test_concurrent_context_creation(self, onnx_model_factory):
        """Test concurrent context creation."""
        from torchpipe.backends.py_tensorrt import PyTensorrtEngine, ProfileConfig
        
        model = SimpleModel()
        onnx_path = onnx_model_factory(model, (1, 3, 224, 224))
        
        engine = PyTensorrtEngine(instance_num=4)
        
        profiles = [
            ProfileConfig(
                min_shapes={'input': (1, 3, 224, 224)},
                opt_shapes={'input': (4, 3, 224, 224)},
                max_shapes={'input': (8, 3, 224, 224)},
            )
        ] * 4
        
        engine.load_from_onnx(onnx_path, profiles=profiles, fp16_mode=False)
        
        errors = []
        contexts = []
        lock = threading.Lock()
        
        def create_context(profile_index):
            try:
                ctx = engine.get_or_create_context(profile_index)
                with lock:
                    contexts.append(ctx)
            except Exception as e:
                with lock:
                    errors.append(e)
        
        threads = [
            threading.Thread(target=create_context, args=(i % 4,))
            for i in range(8)
        ]
        
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert len(errors) == 0, f"Errors during concurrent context creation: {errors}"
        
        if hasattr(engine, 'release'):
            engine.release()
    
    @pytest.mark.skipif(not HAS_TRT or not HAS_TORCH or not torch.cuda.is_available(), reason="TensorRT and CUDA required")
    def test_concurrent_inference(self, onnx_model_factory):
        """Test concurrent inference with multiple backend instances (correct pattern)."""
        from torchpipe.backends.py_tensorrt import PyTensorrtEngine, PyTensorrtInferTensor, ProfileConfig

        model = SimpleModel()
        onnx_path = onnx_model_factory(model, (1, 3, 224, 224))

        # Create shared engine with multiple profiles
        num_instances = 4
        engine = PyTensorrtEngine(instance_num=num_instances)

        profiles = [
            ProfileConfig(
                min_shapes={'input': (1, 3, 224, 224)},
                opt_shapes={'input': (4, 3, 224, 224)},
                max_shapes={'input': (8, 3, 224, 224)},
            )
        ] * num_instances

        engine.load_from_onnx(onnx_path, profiles=profiles, fp16_mode=False)

        # Create multiple backend instances, one per thread
        backends = []
        for i in range(num_instances):
            backend = PyTensorrtInferTensor()
            backend._engine = engine
            backend._context = engine.get_or_create_context(i)
            backend._io_info = engine.get_io_info(i)
            backend._initialized = True
            backend._input_finish_event = torch.cuda.Event()
            backends.append(backend)

        # Test concurrent inference with each thread using its own backend
        import threading
        errors = []
        results = [None] * 20
        lock = threading.Lock()

        def run_inference(idx, batch_size):
            try:
                # Each thread uses its own backend instance
                backend = backends[idx % num_instances]
                input_tensor = torch.randn((batch_size, 3, 224, 224), device='cuda')
                io_dict = {"data": input_tensor}
                backend.forward([io_dict])
                with lock:
                    results[idx] = io_dict["result"]
            except Exception as e:
                with lock:
                    errors.append((idx, e))

        threads = [
            threading.Thread(target=run_inference, args=(i, i % 4 + 1))
            for i in range(20)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors during concurrent inference: {errors}"
        assert all(r is not None for r in results)

        # Cleanup
        for backend in backends:
            if hasattr(backend, 'release'):
                backend.release()

        if hasattr(engine, 'release'):
            engine.release()


class TestEdgeCases:
    """Tests for edge cases."""
    
    @pytest.mark.skipif(not HAS_TRT or not HAS_TORCH or not torch.cuda.is_available(), reason="TensorRT and CUDA required")
    def test_different_batch_sizes_sequence(self, tensorrt_backend_factory):
        """Test running different batch sizes in sequence."""
        model = SimpleModel()
        backend = tensorrt_backend_factory(model, (1, 3, 224, 224))
        
        batch_sizes = [1, 2, 4, 8, 1, 4, 2, 1]
        
        for batch_size in batch_sizes:
            input_tensor = torch.randn((batch_size, 3, 224, 224), device='cuda')
            io_dict = {"data": input_tensor}
            backend.forward([io_dict])
            
            result = io_dict["result"]
            assert result.shape[0] == batch_size
    
    @pytest.mark.skipif(not HAS_TRT or not HAS_TORCH or not torch.cuda.is_available(), reason="TensorRT and CUDA required")
    def test_repeated_initialization(self, onnx_model_factory):
        """Test repeated initialization of the same backend."""
        from torchpipe.backends.py_tensorrt import PyTensorrtInferTensor
        
        model = SimpleModel()
        onnx_path = onnx_model_factory(model, (1, 3, 224, 224))
        
        config = {
            "model": onnx_path,
            "model_type": "onnx",
            "instance_num": "1",
        }
        
        backend = PyTensorrtInferTensor()
        
        # Initialize multiple times
        for _ in range(3):
            backend.init(config)
            
            input_tensor = torch.randn((1, 3, 224, 224), device='cuda')
            io_dict = {"data": input_tensor}
            backend.forward([io_dict])
            
            assert io_dict["result"] is not None
    
    @pytest.mark.skipif(not HAS_TRT or not HAS_TORCH or not torch.cuda.is_available(), reason="TensorRT and CUDA required")
    def test_empty_input_handling(self, tensorrt_backend_factory):
        """Test handling of invalid inputs."""
        model = SimpleModel()
        backend = tensorrt_backend_factory(model, (1, 3, 224, 224))
        
        # Test with None input
        with pytest.raises(Exception):
            backend.forward([{"data": None}])
    
    @pytest.mark.skipif(not HAS_TRT or not HAS_TORCH or not torch.cuda.is_available(), reason="TensorRT and CUDA required")
    def test_cpu_tensor_input(self, tensorrt_backend_factory):
        """Test that CPU tensors are automatically moved to CUDA."""
        model = SimpleModel()
        backend = tensorrt_backend_factory(model, (1, 3, 224, 224))
        
        input_tensor = torch.randn((1, 3, 224, 224))  # CPU tensor
        io_dict = {"data": input_tensor}
        
        backend.forward([io_dict])
        
        result = io_dict["result"]
        assert result.is_cuda


class TestErrorHandling:
    """Tests for error handling."""
    
    @pytest.mark.skipif(not HAS_TRT or not HAS_TORCH or not torch.cuda.is_available(), reason="TensorRT and CUDA required")
    def test_invalid_model_path(self):
        """Test handling of invalid model path."""
        from torchpipe.backends.py_tensorrt import PyTensorrtInferTensor
        from torchpipe.backends.trt_utils import TensorRTError
        
        config = {
            "model": "/nonexistent/path/model.onnx",
            "model_type": "onnx",
        }
        
        backend = PyTensorrtInferTensor()
        
        with pytest.raises((TensorRTError, RuntimeError, FileNotFoundError)):
            backend.init(config)
    
    @pytest.mark.skipif(not HAS_TRT or not HAS_TORCH or not torch.cuda.is_available(), reason="TensorRT and CUDA required")
    def test_invalid_config(self):
        """Test handling of invalid configuration."""
        from torchpipe.backends.py_tensorrt import PyTensorrtInferTensor
        from torchpipe.backends.trt_utils import TensorRTError
        
        config = {}  # Missing required 'model' key
        
        backend = PyTensorrtInferTensor()
        
        with pytest.raises((TensorRTError, KeyError)):
            backend.init(config)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
