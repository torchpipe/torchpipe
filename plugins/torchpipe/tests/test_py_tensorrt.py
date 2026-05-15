"""
Tests for PyTensorrtTensor and related components.

This module provides comprehensive tests for the Python TensorRT backend
implementation, following the patterns from test_trt.py and test_v2_trt.py.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from typing import Tuple, List
from unittest import mock

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
except (ImportError, OSError) as e:
    HAS_TRT = False

try:
    import omniback
    HAS_OMNIBACK = True
except ImportError:
    HAS_OMNIBACK = False

try:
    import tvm_ffi
    HAS_TVM_FFI = True
except ImportError:
    HAS_TVM_FFI = False


def pytest_skip_if_no_cuda():
    """Skip test if CUDA is not available."""
    if not HAS_TORCH or not torch.cuda.is_available():
        pytest.skip("CUDA not available")


def pytest_skip_if_no_trt():
    """Skip test if TensorRT is not available."""
    if not HAS_TRT:
        pytest.skip("TensorRT not available")


class Identity(torch.nn.Module):
    """Simple identity model that multiplies input by 2."""
    
    def forward(self, x):
        return x * 2


class Conv(torch.nn.Module):
    """Simple Conv2d model for testing."""
    
    def __init__(self):
        super(Conv, self).__init__()
        self.conv = torch.nn.Conv2d(3, 1, kernel_size=3, stride=2, padding=1)
    
    def forward(self, x):
        return self.conv(x)


class MultiInput(torch.nn.Module):
    """Multi-input model for testing."""
    
    def forward(self, x, y):
        return x + y


def get_tmp_onnx(model: torch.nn.Module, input_shape, onnx_path: str = None) -> str:
    """
    Export PyTorch model to ONNX format.
    
    Args:
        model: PyTorch model to export
        input_shape: Input tensor shape (can be list or tuple of shapes for multi-input)
        onnx_path: Optional path for ONNX file
    
    Returns:
        Path to temporary ONNX file
    """
    if onnx_path is None:
        onnx_path = tempfile.mktemp(suffix=".onnx")
    
    model.eval()
    
    if isinstance(input_shape[0], (list, tuple)):
        input_data = [torch.randn(shape) for shape in input_shape]
        input_names = [f"input_{i}" for i in range(len(input_shape))]
        dynamic_axes = {name: {0: "batch_size"} for name in input_names}
    else:
        input_data = torch.randn(input_shape)
        input_names = ["input"]
        dynamic_axes = {"input": {0: "batch_size"}}
    
    torch.onnx.export(
        model,
        input_data if isinstance(input_data, torch.Tensor) else tuple(input_data),
        onnx_path,
        input_names=input_names,
        output_names=["output"],
        dynamic_axes=dynamic_axes
    )
    return onnx_path


def get_tmp_onnx_cuda(model: torch.nn.Module, input_shape, dtype=torch.float16, onnx_path: str = None) -> str:
    """
    Export PyTorch model to ONNX format with CUDA tensor.
    
    Args:
        model: PyTorch model to export
        input_shape: Input tensor shape
        dtype: Data type for input tensor
        onnx_path: Optional path for ONNX file
    
    Returns:
        Path to temporary ONNX file
    """
    if onnx_path is None:
        onnx_path = tempfile.mktemp(suffix=".onnx")
    
    model.eval()
    input_data = torch.randn(input_shape).cuda().to(dtype)
    
    torch.onnx.export(
        model,
        input_data,
        onnx_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}}
    )
    return onnx_path


# =============================================================================
# Unit Tests for CUDA Utilities
# =============================================================================

class TestCUDAStreamManager:
    """Tests for CUDAStreamManager."""
    
    @pytest.mark.skipif(not HAS_TORCH or not torch.cuda.is_available(), reason="CUDA not available")
    def test_is_available(self):
        """Test CUDA availability check."""
        from torchpipe.backends.cuda_utils import CUDAStreamManager
        
        assert CUDAStreamManager.is_available() == torch.cuda.is_available()
    
    @pytest.mark.skipif(not HAS_TORCH or not torch.cuda.is_available(), reason="CUDA not available")
    def test_get_current_stream(self):
        """Test getting current stream."""
        from torchpipe.backends.cuda_utils import CUDAStreamManager
        
        stream = CUDAStreamManager.get_current_stream()
        assert stream is not None
        assert isinstance(stream, torch.cuda.Stream)
    
    @pytest.mark.skipif(not HAS_TORCH or not torch.cuda.is_available(), reason="CUDA not available")
    def test_get_default_stream(self):
        """Test getting default stream."""
        from torchpipe.backends.cuda_utils import CUDAStreamManager
        
        stream = CUDAStreamManager.get_default_stream()
        assert stream is not None
        assert isinstance(stream, torch.cuda.Stream)
    
    @pytest.mark.skipif(not HAS_TORCH or not torch.cuda.is_available(), reason="CUDA not available")
    def test_create_stream(self):
        """Test creating a new stream."""
        from torchpipe.backends.cuda_utils import CUDAStreamManager
        
        stream = CUDAStreamManager.create_stream()
        assert stream is not None
        assert isinstance(stream, torch.cuda.Stream)
        
        high_priority_stream = CUDAStreamManager.create_stream(high_priority=True)
        assert high_priority_stream is not None
    
    @pytest.mark.skipif(not HAS_TORCH or not torch.cuda.is_available(), reason="CUDA not available")
    def test_create_event(self):
        """Test creating a CUDA event."""
        from torchpipe.backends.cuda_utils import CUDAStreamManager
        
        event = CUDAStreamManager.create_event()
        assert event is not None
        assert isinstance(event, torch.cuda.Event)
    
    @pytest.mark.skipif(not HAS_TORCH or not torch.cuda.is_available(), reason="CUDA not available")
    def test_record_and_wait_event(self):
        """Test recording and waiting for events."""
        from torchpipe.backends.cuda_utils import CUDAStreamManager
        
        stream1 = CUDAStreamManager.create_stream()
        stream2 = CUDAStreamManager.create_stream()
        
        event = CUDAStreamManager.record_event(stream1)
        assert event is not None
        
        CUDAStreamManager.wait_event(stream2, event)
    
    @pytest.mark.skipif(not HAS_TORCH or not torch.cuda.is_available(), reason="CUDA not available")
    def test_is_using_default_stream(self):
        """Test checking if using default stream."""
        from torchpipe.backends.cuda_utils import CUDAStreamManager
        
        result = CUDAStreamManager.is_using_default_stream()
        assert isinstance(result, bool)
    
    @pytest.mark.skipif(not HAS_TORCH or not torch.cuda.is_available(), reason="CUDA not available")
    def test_sync_streams(self):
        """Test synchronizing two streams."""
        from torchpipe.backends.cuda_utils import CUDAStreamManager
        
        stream1 = CUDAStreamManager.create_stream()
        stream2 = CUDAStreamManager.create_stream()
        
        event = CUDAStreamManager.sync_streams(stream1, stream2)
        assert event is not None


class TestStreamPool:
    """Tests for StreamPool."""
    
    @pytest.mark.skipif(not HAS_TORCH or not torch.cuda.is_available(), reason="CUDA not available")
    def test_create_pool(self):
        """Test creating a stream pool."""
        from torchpipe.backends.cuda_utils import StreamPool
        
        pool = StreamPool(num_streams=4)
        assert pool.num_streams == 4
        assert pool.available_count == 4
    
    @pytest.mark.skipif(not HAS_TORCH or not torch.cuda.is_available(), reason="CUDA not available")
    def test_acquire_and_release(self):
        """Test acquiring and releasing streams."""
        from torchpipe.backends.cuda_utils import StreamPool
        
        pool = StreamPool(num_streams=2)
        
        index1, se1 = pool.acquire()
        assert index1 in [0, 1]
        assert pool.available_count == 1
        
        index2, se2 = pool.acquire()
        assert pool.available_count == 0
        
        pool.release(index1)
        assert pool.available_count == 1
        
        pool.release(index2)
        assert pool.available_count == 2
    
    @pytest.mark.skipif(not HAS_TORCH or not torch.cuda.is_available(), reason="CUDA not available")
    def test_use_stream_context_manager(self):
        """Test using stream context manager."""
        from torchpipe.backends.cuda_utils import StreamPool
        
        pool = StreamPool(num_streams=2)
        
        with pool.use_stream() as (index, stream_event):
            assert index in [0, 1]
            assert stream_event.stream is not None
        
        assert pool.available_count == 2


# =============================================================================
# Unit Tests for TensorRT Utilities
# =============================================================================

class TestTRTUtils:
    """Tests for TensorRT utilities."""
    
    def test_dims64_from_tuple(self):
        """Test Dims64 creation from tuple."""
        from torchpipe.backends.trt_utils import Dims64
        
        dims = Dims64.from_tuple((1, 2, 3))
        assert dims.nbDims == 3
        assert dims.to_tuple() == (1, 2, 3)
    
    def test_dims64_iteration(self):
        """Test Dims64 iteration."""
        from torchpipe.backends.trt_utils import Dims64
        
        dims = Dims64.from_tuple((1, 2, 3))
        assert list(dims) == [1, 2, 3]
    
    @pytest.mark.skipif(not HAS_TRT, reason="TensorRT not available")
    def test_datatype_conversion(self):
        """Test DataType conversions."""
        from torchpipe.backends.trt_utils import (
            DataType,
            trt_dtype_to_datatype,
            datatype_to_torch_dtype,
            torch_dtype_to_datatype,
        )
        
        assert trt_dtype_to_datatype(trt.DataType.FLOAT) == DataType.FP32
        assert trt_dtype_to_datatype(trt.DataType.HALF) == DataType.FP16
        assert trt_dtype_to_datatype(trt.DataType.INT32) == DataType.INT32
    
    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
    def test_datatype_to_torch_dtype(self):
        """Test DataType to torch dtype conversion."""
        from torchpipe.backends.trt_utils import DataType, datatype_to_torch_dtype
        
        assert datatype_to_torch_dtype(DataType.FP32) == torch.float32
        assert datatype_to_torch_dtype(DataType.FP16) == torch.float16
        assert datatype_to_torch_dtype(DataType.INT32) == torch.int32
    
    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
    def test_torch_dtype_to_datatype(self):
        """Test torch dtype to DataType conversion."""
        from torchpipe.backends.trt_utils import torch_dtype_to_datatype, DataType
        
        assert torch_dtype_to_datatype(torch.float32) == DataType.FP32
        assert torch_dtype_to_datatype(torch.float16) == DataType.FP16
    
    def test_element_size(self):
        """Test element size calculation."""
        from torchpipe.backends.trt_utils import DataType, element_size
        
        assert element_size(DataType.INT8) == 1
        assert element_size(DataType.FP16) == 2
        assert element_size(DataType.FP32) == 4
        assert element_size(DataType.INT64) == 8
    
    def test_match_shape(self):
        """Test shape matching."""
        from torchpipe.backends.trt_utils import match_shape
        
        assert match_shape((1, 2, 3), (1, 2, 3))
        assert match_shape((1, 2, 3), (-1, 2, 3))
        assert match_shape((1, 2, 3), (1, -1, 3))
        assert not match_shape((1, 2, 3), (1, 2, 4))
        assert not match_shape((1, 2, 3), (1, 2))


# =============================================================================
# Unit Tests for Backend Base
# =============================================================================

class TestBackendBase:
    """Tests for BackendBase."""
    
    def test_backend_base_init(self):
        """Test BackendBase initialization."""
        from torchpipe.backends.base import BackendBase
        
        backend = BackendBase()
        assert not backend.is_initialized()
        
        backend.init({"key": "value"})
        assert backend.is_initialized()
        assert backend._config == {"key": "value"}
    
    def test_backend_base_forward_not_implemented(self):
        """Test that forward raises NotImplementedError."""
        from torchpipe.backends.base import BackendBase
        
        backend = BackendBase()
        backend.init({})
        
        with pytest.raises(NotImplementedError):
            backend.forward([{}])
    
    def test_backend_base_max_min(self):
        """Test default max/min values."""
        from torchpipe.backends.base import BackendBase
        
        backend = BackendBase()
        assert backend.max() == 1
        assert backend.min() == 1


class TestBackendRegistration:
    """Tests for backend registration."""
    
    def test_register_backend_decorator(self):
        """Test backend registration with decorator."""
        from torchpipe.backends.base import register_backend, get_backend, BackendMeta
        
        @register_backend("TestBackend1", meta=BackendMeta(name="TestBackend1"))
        class TestBackend1:
            def init(self, config, options=None): pass
            def forward(self, ios): pass
            def max(self): return 1
            def min(self): return 1
        
        assert get_backend("TestBackend1") is not None
    
    def test_backend_decorator(self):
        """Test @backend decorator."""
        from torchpipe.backends.base import backend, get_backend, BackendMeta
        
        @backend("TestBackend2", BackendMeta(name="TestBackend2"))
        class TestBackend2:
            def init(self, config, options=None): pass
            def forward(self, ios): pass
            def max(self): return 1
            def min(self): return 1
        
        assert get_backend("TestBackend2") is not None
    
    def test_list_backends(self):
        """Test listing backends."""
        from torchpipe.backends.base import list_backends
        
        backends = list_backends()
        assert isinstance(backends, list)


# =============================================================================
# Unit Tests for PyTensorrtEngine
# =============================================================================

class TestPyTensorrtEngine:
    """Tests for PyTensorrtEngine."""
    
    @pytest.mark.skipif(not HAS_TRT, reason="TensorRT not available")
    def test_engine_creation(self):
        """Test that PyTensorrtEngine can be instantiated."""
        from torchpipe.backends.py_tensorrt import PyTensorrtEngine
        
        engine = PyTensorrtEngine()
        assert engine.engine is None
        assert engine.io_info is None
    
    @pytest.mark.skipif(not HAS_TRT or not HAS_TORCH or not torch.cuda.is_available(), reason="TensorRT and CUDA required")
    def test_load_onnx_and_build_engine(self):
        """Test loading ONNX and building TensorRT engine."""
        from torchpipe.backends.py_tensorrt import PyTensorrtEngine, ProfileConfig
        
        tmp_onnx = get_tmp_onnx(Identity(), [1, 3, 224, 224])
        
        try:
            engine = PyTensorrtEngine(instance_num=1)
            
            profiles = [
                ProfileConfig(
                    min_shapes={'input': (1, 3, 224, 224)},
                    opt_shapes={'input': (4, 3, 224, 224)},
                    max_shapes={'input': (8, 3, 224, 224)},
                )
            ]
            
            engine.load_from_onnx(
                tmp_onnx,
                profiles=profiles,
                fp16_mode=False
            )
            
            assert engine.engine is not None
            assert engine.io_info is not None
            
            context = engine.create_context()
            assert context is not None
        finally:
            if os.path.exists(tmp_onnx):
                os.remove(tmp_onnx)


# =============================================================================
# Integration Tests for PyTensorrtInferTensor (similar to test_trt.py)
# =============================================================================

class TestPyTensorrtInferTensor:
    """Tests for PyTensorrtInferTensor - following test_trt.py patterns."""
    
    @pytest.fixture
    def identity_model_config(self):
        """Fixture to create Identity model configuration."""
        tmp_onnx = get_tmp_onnx(Identity(), [1, 3, 224, 224])
        
        config = {
            "model": tmp_onnx,
            "model_type": "onnx",
            "instance_num": "1",
            "instance_index": "0",
        }
        
        yield config, tmp_onnx
        
        if os.path.exists(tmp_onnx):
            os.remove(tmp_onnx)
    
    @pytest.fixture
    def conv_model_config(self):
        """Fixture to create Conv model configuration (FP16)."""
        torch_model = Conv().cuda().half()
        tmp_onnx = get_tmp_onnx_cuda(torch_model, [1, 3, 224, 224])
        
        config = {
            "model": tmp_onnx,
            "model_type": "onnx",
            "instance_num": "1",
            "instance_index": "0",
        }
        
        yield config, torch_model, tmp_onnx
        
        if os.path.exists(tmp_onnx):
            os.remove(tmp_onnx)
    
    @pytest.mark.skipif(not HAS_TRT or not HAS_TORCH or not torch.cuda.is_available(), reason="TensorRT and CUDA required")
    def test_backend_initialization(self, identity_model_config):
        """Test backend initialization."""
        from torchpipe.backends.py_tensorrt import PyTensorrtInferTensor
        
        config, _ = identity_model_config
        
        backend = PyTensorrtInferTensor()
        backend.init(config)
        
        assert backend.is_initialized()
        assert backend._engine is not None
        assert backend._context is not None
    
    @pytest.mark.skipif(not HAS_TRT or not HAS_TORCH or not torch.cuda.is_available(), reason="TensorRT and CUDA required")
    def test_forward_single_input(self, identity_model_config):
        """Test forward with single input - similar to test_trt.py."""
        from torchpipe.backends.py_tensorrt import PyTensorrtInferTensor
        
        config, _ = identity_model_config
        
        backend = PyTensorrtInferTensor()
        backend.init(config)
        
        input_tensor = torch.ones((1, 3, 224, 224), device='cuda')
        data = {"data": input_tensor}
        
        backend.forward([data])
        
        result = data['result']
        assert isinstance(result, torch.Tensor)
        
        expected = input_tensor * 2
        assert torch.allclose(result, expected, rtol=1e-2, atol=1e-2)
    
    @pytest.mark.skipif(not HAS_TRT or not HAS_TORCH or not torch.cuda.is_available(), reason="TensorRT and CUDA required")
    def test_forward_batch_input(self, identity_model_config):
        """Test forward with batch input."""
        from torchpipe.backends.py_tensorrt import PyTensorrtInferTensor
        
        config, _ = identity_model_config
        
        backend = PyTensorrtInferTensor()
        backend.init(config)
        
        batch_size = 4
        input_tensor = torch.ones((batch_size, 3, 224, 224), device='cuda')
        data = {"data": input_tensor}
        
        backend.forward([data])
        
        result = data['result']
        assert result.shape[0] == batch_size
        
        expected = input_tensor * 2
        assert torch.allclose(result, expected, rtol=1e-2, atol=1e-2)
    
    @pytest.mark.skipif(not HAS_TRT or not HAS_TORCH or not torch.cuda.is_available(), reason="TensorRT and CUDA required")
    def test_forward_conv_model_fp16(self, conv_model_config):
        """Test forward with Conv model in FP16 - similar to test_v2_trt.py."""
        from torchpipe.backends.py_tensorrt import PyTensorrtInferTensor
        
        config, torch_model, _ = conv_model_config
        
        backend = PyTensorrtInferTensor()
        backend.init(config)
        
        input_tensor = torch.ones((1, 3, 224, 224), dtype=torch.float16, device='cuda') * 10
        data = {"data": input_tensor}
        
        backend.forward([data])
        
        result = data['result']
        expected = torch_model(input_tensor)
        
        assert torch.allclose(result, expected, rtol=1e-2, atol=1e-2)
    
    @pytest.mark.skipif(not HAS_TRT or not HAS_TORCH or not torch.cuda.is_available(), reason="TensorRT and CUDA required")
    def test_max_min(self, identity_model_config):
        """Test max/min methods."""
        from torchpipe.backends.py_tensorrt import PyTensorrtInferTensor
        
        config, _ = identity_model_config
        
        backend = PyTensorrtInferTensor()
        backend.init(config)
        
        max_bs = backend.max()
        min_bs = backend.min()
        
        assert isinstance(max_bs, int)
        assert isinstance(min_bs, int)
        assert max_bs >= min_bs


# =============================================================================
# Integration Tests with omniback (similar to test_trt.py patterns)
# =============================================================================

@pytest.mark.skipif(not HAS_OMNIBACK, reason="omniback not available")
class TestPyTensorrtWithOmniback:
    """Tests for PyTensorrtInferTensor with omniback integration."""
    
    @pytest.fixture
    def model_config(self):
        """Fixture to create model configuration."""
        tmp_onnx = get_tmp_onnx(Identity(), [1, 3, 224, 224])
        
        config = {
            "model": tmp_onnx,
            "model_type": "onnx",
        }
        
        yield config, tmp_onnx
        
        if os.path.exists(tmp_onnx):
            os.remove(tmp_onnx)
    
    @pytest.mark.skipif(not HAS_TRT or not HAS_TORCH or not torch.cuda.is_available(), reason="TensorRT and CUDA required")
    def test_init_with_omniback(self, model_config):
        """Test initializing backend via omniback.init()."""
        config, _ = model_config
        
        model = omniback.init("PyTensorrtInferTensor", config)
        
        input_tensor = torch.ones((1, 3, 224, 224))
        data = omniback.Dict({"data": input_tensor})
        
        model([data])
        
        result = data['result']
        if not isinstance(result, torch.Tensor):
            result = torch.from_dlpack(result)
        
        expected = input_tensor.cuda() * 2
        assert torch.allclose(result, expected, rtol=1e-2, atol=1e-2)


# =============================================================================
# Tests for PySyncTensor
# =============================================================================

class TestPySyncTensor:
    """Tests for PySyncTensor."""
    
    @pytest.mark.skipif(not HAS_TORCH or not torch.cuda.is_available(), reason="CUDA not available")
    def test_sync_tensor_requires_task_index(self):
        """Test that SyncTensor requires TASK_INDEX_KEY."""
        from torchpipe.backends.py_sync import PySyncTensor
        
        sync = PySyncTensor()
        
        with pytest.raises(RuntimeError) as exc_info:
            sync.init({})
        
        assert "independent thread mode" in str(exc_info.value)
    
    @pytest.mark.skipif(not HAS_TORCH or not torch.cuda.is_available(), reason="CUDA not available")
    def test_sync_tensor_initialization(self):
        """Test SyncTensor initialization with TASK_INDEX_KEY."""
        from torchpipe.backends.py_sync import PySyncTensor
        
        sync = PySyncTensor()
        config = {"instance_index": "0"}
        
        sync.init(config)
        
        assert sync.is_initialized()
        assert sync._independent_thread_index == 0
    
    @pytest.mark.skipif(not HAS_TORCH or not torch.cuda.is_available(), reason="CUDA not available")
    def test_sync_tensor_forward(self):
        """Test SyncTensor forward execution."""
        from torchpipe.backends.py_sync import PySyncTensor
        
        sync = PySyncTensor()
        config = {"instance_index": "0"}
        sync.init(config)
        
        class MockBackend:
            def __init__(self):
                self.forward_called = False
            
            def forward(self, ios):
                self.forward_called = True
                ios[0]["result"] = "test_result"
        
        mock_backend = MockBackend()
        sync.set_owned_backend(mock_backend)
        
        io_dict = {}
        sync.forward([io_dict])
        
        assert mock_backend.forward_called
        assert io_dict["result"] == "test_result"


# =============================================================================
# Performance Tests (similar to test_bench_trt.py patterns)
# =============================================================================

class TestPerformance:
    """Performance tests for PyTensorrtInferTensor."""
    
    @pytest.fixture
    def perf_model_config(self):
        """Fixture to create model configuration for performance testing."""
        torch_model = Conv().cuda().half()
        tmp_onnx = get_tmp_onnx_cuda(torch_model, [1, 3, 224, 224])
        
        config = {
            "model": tmp_onnx,
            "model_type": "onnx",
        }
        
        yield config, torch_model, tmp_onnx
        
        if os.path.exists(tmp_onnx):
            os.remove(tmp_onnx)
    
    @pytest.mark.skipif(not HAS_TRT or not HAS_TORCH or not torch.cuda.is_available(), reason="TensorRT and CUDA required")
    def test_forward_latency(self, perf_model_config):
        """Test forward latency."""
        import time
        
        from torchpipe.backends.py_tensorrt import PyTensorrtInferTensor
        
        config, torch_model, _ = perf_model_config
        
        backend = PyTensorrtInferTensor()
        backend.init(config)
        
        input_tensor = torch.ones((1, 3, 224, 224), dtype=torch.float16, device='cuda')
        
        for _ in range(10):
            io_dict = {"data": input_tensor}
            backend.forward([io_dict])
        
        torch.cuda.synchronize()
        
        num_iterations = 100
        start = time.perf_counter()
        
        for _ in range(num_iterations):
            io_dict = {"data": input_tensor}
            backend.forward([io_dict])
        
        torch.cuda.synchronize()
        end = time.perf_counter()
        
        avg_latency_ms = (end - start) / num_iterations * 1000
        
        print(f"\nAverage forward latency: {avg_latency_ms:.3f} ms")
        
        assert avg_latency_ms < 100
    
    @pytest.mark.skipif(not HAS_TRT or not HAS_TORCH or not torch.cuda.is_available(), reason="TensorRT and CUDA required")
    def test_throughput(self, perf_model_config):
        """Test throughput with different batch sizes."""
        import time
        
        from torchpipe.backends.py_tensorrt import PyTensorrtInferTensor
        
        config, _, _ = perf_model_config
        
        backend = PyTensorrtInferTensor()
        backend.init(config)
        
        for batch_size in [1, 2, 4]:
            input_tensor = torch.ones((batch_size, 3, 224, 224), dtype=torch.float16, device='cuda')
            
            for _ in range(5):
                io_dict = {"data": input_tensor}
                backend.forward([io_dict])
            
            torch.cuda.synchronize()
            
            num_iterations = 50
            start = time.perf_counter()
            
            for _ in range(num_iterations):
                io_dict = {"data": input_tensor}
                backend.forward([io_dict])
            
            torch.cuda.synchronize()
            end = time.perf_counter()
            
            total_samples = batch_size * num_iterations
            throughput = total_samples / (end - start)
            
            print(f"\nBatch size {batch_size}: {throughput:.1f} samples/sec")


# =============================================================================
# DLPack Interop Tests
# =============================================================================

class TestDLPackInterop:
    """Tests for DLPack interoperability."""
    
    @pytest.mark.skipif(not HAS_TVM_FFI or not HAS_TORCH or not torch.cuda.is_available(), reason="tvm_ffi and CUDA required")
    def test_dlpack_roundtrip(self):
        """Test DLPack roundtrip conversion."""
        import tvm_ffi
        
        original = torch.randn(10, 10, device='cuda')
        
        tvm_tensor = tvm_ffi.from_dlpack(original, require_contiguous=True)
        
        torch_tensor = torch.from_dlpack(tvm_tensor)
        
        assert torch.allclose(original, torch_tensor)
    
    @pytest.mark.skipif(not HAS_TORCH or not torch.cuda.is_available() or not HAS_TRT, reason="CUDA and TensorRT required")
    def test_dlpack_with_backend(self):
        """Test DLPack with backend input."""
        from torchpipe.backends.py_tensorrt import PyTensorrtInferTensor
        
        tmp_onnx = get_tmp_onnx(Identity(), [1, 3, 224, 224])
        
        try:
            config = {
                "model": tmp_onnx,
                "model_type": "onnx",
                "instance_num": "1",
                "instance_index": "0",
            }
            
            backend = PyTensorrtInferTensor()
            backend.init(config)
            
            input_tensor = torch.ones((1, 3, 224, 224), device='cuda')
            
            if HAS_OMNIBACK:
                io_dict = omniback.Dict({"data": input_tensor})
            else:
                io_dict = {"data": input_tensor}
            
            backend.forward([io_dict])
            
            result = io_dict["result"]
            # Result may be a tvm_ffi.Tensor when using omniback.Dict
            # Convert to torch.Tensor if needed
            if not isinstance(result, torch.Tensor):
                if hasattr(result, '__dlpack__'):
                    result = torch.from_dlpack(result)
            assert isinstance(result, torch.Tensor)
            
        finally:
            if os.path.exists(tmp_onnx):
                os.remove(tmp_onnx)


# =============================================================================
# Multi-Profile Tests
# =============================================================================

class TestMultiProfile:
    """Tests for multi-profile support."""
    
    @pytest.mark.skipif(not HAS_TRT or not HAS_TORCH or not torch.cuda.is_available(), reason="TensorRT and CUDA required")
    def test_dynamic_batch_with_profile(self):
        """Test dynamic batch size with optimization profile."""
        from torchpipe.backends.py_tensorrt import PyTensorrtEngine, PyTensorrtInferTensor, ProfileConfig
        
        tmp_onnx = get_tmp_onnx(Identity(), [1, 3, 224, 224])
        
        try:
            engine = PyTensorrtEngine(instance_num=1)
            
            profiles = [
                ProfileConfig(
                    min_shapes={'input': (1, 3, 224, 224)},
                    opt_shapes={'input': (4, 3, 224, 224)},
                    max_shapes={'input': (8, 3, 224, 224)},
                )
            ]
            
            engine.load_from_onnx(
                tmp_onnx,
                profiles=profiles,
                fp16_mode=False
            )
            
            backend = PyTensorrtInferTensor()
            backend._engine = engine
            backend._context = engine.get_or_create_context(0)
            backend._io_info = engine.get_io_info(0)
            backend._initialized = True
            backend._input_finish_event = torch.cuda.Event()
            
            for batch_size in [1, 2, 4]:
                input_tensor = torch.ones((batch_size, 3, 224, 224), device='cuda')
                io_dict = {"data": input_tensor}
                backend.forward([io_dict])
                
                result = io_dict["result"]
                assert result.shape[0] == batch_size
                
                expected = input_tensor * 2
                assert torch.allclose(result, expected, rtol=1e-2, atol=1e-2)
            
        finally:
            if os.path.exists(tmp_onnx):
                os.remove(tmp_onnx)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
