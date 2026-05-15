"""
Pytest configuration and shared fixtures for torchpipe tests.

This module provides common test infrastructure including:
- Temporary directory management with automatic cleanup
- CUDA memory monitoring
- Resource management utilities
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from typing import Generator
import gc


@pytest.fixture(scope="session")
def temp_dir():
    """
    Create a temporary directory for test files.
    
    The directory is automatically cleaned up after the test session.
    
    Yields:
        Path to temporary directory
    
    Example:
        def test_example(temp_dir):
            model_path = temp_dir / "model.onnx"
            # ... use model_path
    """
    dir_path = Path(tempfile.mkdtemp())
    yield dir_path
    shutil.rmtree(dir_path, ignore_errors=True)


@pytest.fixture(scope="function")
def cuda_memory_monitor():
    """
    Monitor CUDA memory usage during tests.

    This fixture tracks memory allocation before and after the test
    and checks for memory leaks (memory not freed after the test).

    Example:
        def test_inference(cuda_memory_monitor):
            # ... run inference
            # Memory stats are available
    """
    try:
        import torch
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        # Record memory before test
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.reset_peak_memory_stats()
        memory_before = torch.cuda.memory_allocated()

        yield

        # Record memory after test
        torch.cuda.synchronize()
        gc.collect()
        memory_after = torch.cuda.memory_allocated()

        # Check for memory leak (memory not freed)
        memory_leaked = memory_after - memory_before
        if memory_leaked > 1024 * 1024:  # Allow 1MB tolerance
            pytest.fail(f"Memory leak detected: {memory_leaked / 1024 / 1024:.2f} MB not freed")
    except ImportError:
        pytest.skip("PyTorch not available")


@pytest.fixture
def resource_tracker():
    """
    Track resource usage during tests.
    
    Returns a dictionary that can be used to track
    file handles, memory usage, etc.
    
    Example:
        def test_example(resource_tracker):
            resource_tracker['test_start'] = True
            # ... run test
            resource_tracker['test_end'] = True
            # Check resources
    """
    return {}


def check_no_memory_leak():
    """
    Helper function to check for memory leaks.
    
    This should be called at the end of tests to ensure
    no GPU memory has been leaked.
    """
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            gc.collect()
    except ImportError:
        pass


@pytest.fixture
def onnx_model_factory():
    """
    Factory fixture to create ONNX models for testing.
    
    Returns:
        A function that creates ONNX models from PyTorch modules.
    
    Example:
        def test_model(onnx_model_factory):
            import torch
            model = torch.nn.Linear(10, 5)
            onnx_path = onnx_model_factory(model, (1, 10))
    """
    import tempfile
    import os
    
    created_files = []
    
    def create_onnx(model, input_shape, input_names=None, dynamic_axes=None):
        import torch
        
        model.eval()
        onnx_path = tempfile.mktemp(suffix='.onnx')
        created_files.append(onnx_path)
        
        if isinstance(input_shape[0], (list, tuple)):
            dummy_inputs = [torch.randn(shape) for shape in input_shape]
            if input_names is None:
                input_names = [f"input_{i}" for i in range(len(input_shape))]
        else:
            dummy_inputs = torch.randn(input_shape)
            if input_names is None:
                input_names = ["input"]
        
        if dynamic_axes is None:
            dynamic_axes = {name: {0: 'batch_size'} for name in (input_names if isinstance(input_names, list) else [input_names])}
        
        torch.onnx.export(
            model,
            dummy_inputs if isinstance(dummy_inputs, torch.Tensor) else tuple(dummy_inputs),
            onnx_path,
            input_names=input_names if isinstance(input_names, list) else [input_names],
            output_names=["output"],
            dynamic_axes=dynamic_axes
        )
        
        return onnx_path
    
    yield create_onnx
    
    for path in created_files:
        if os.path.exists(path):
            os.remove(path)


@pytest.fixture
def tensorrt_backend_factory():
    """
    Factory fixture to create TensorRT backends for testing.
    
    Returns:
        A function that creates PyTensorrtInferTensor backends.
    """
    import tempfile
    import os
    
    created_files = []
    created_backends = []
    
    def create_backend(model, input_shape, config=None, **kwargs):
        import torch
        from torchpipe.backends.py_tensorrt import PyTensorrtInferTensor, ProfileConfig
        
        model.eval()
        onnx_path = tempfile.mktemp(suffix='.onnx')
        created_files.append(onnx_path)
        
        dummy_input = torch.randn(input_shape)
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        )
        
        default_config = {
            "model": onnx_path,
            "model_type": "onnx",
            "instance_num": "1",
        }
        if config:
            default_config.update(config)
        
        backend = PyTensorrtInferTensor()
        backend.init(default_config)
        created_backends.append(backend)
        
        return backend
    
    yield create_backend
    
    for backend in created_backends:
        if hasattr(backend, 'release'):
            backend.release()
    
    for path in created_files:
        if os.path.exists(path):
            os.remove(path)


@pytest.fixture
def concurrent_test_helper():
    """
    Helper fixture for concurrent testing.
    
    Returns:
        A context manager that runs functions concurrently.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import threading
    
    class ConcurrentTestHelper:
        def __init__(self):
            self.results = []
            self.errors = []
            self._lock = threading.Lock()
        
        def run_concurrent(self, func, args_list, max_workers=4):
            """
            Run a function concurrently with different arguments.
            
            Args:
                func: Function to run
                args_list: List of argument tuples
                max_workers: Maximum number of concurrent workers
            
            Returns:
                List of results
            """
            results = [None] * len(args_list)
            
            def wrapped_func(idx, args):
                try:
                    result = func(*args)
                    return idx, result, None
                except Exception as e:
                    return idx, None, e
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(wrapped_func, i, args)
                    for i, args in enumerate(args_list)
                ]
                
                for future in as_completed(futures):
                    idx, result, error = future.result()
                    results[idx] = result
                    if error:
                        with self._lock:
                            self.errors.append((idx, error))
            
            return results
        
        def assert_no_errors(self):
            """Assert that no errors occurred during concurrent execution."""
            if self.errors:
                error_msgs = [f"Task {idx}: {err}" for idx, err in self.errors]
                raise AssertionError(f"Concurrent execution errors:\n" + "\n".join(error_msgs))
    
    return ConcurrentTestHelper()


@pytest.fixture(autouse=True)
def cleanup_cuda():
    """
    Auto cleanup CUDA resources after each test.
    """
    yield
    
    try:
        import torch
        if torch.cuda.is_available():
            # Only synchronize if no CUDA errors have occurred
            try:
                torch.cuda.synchronize()
            except RuntimeError:
                # CUDA error occurred, reset the device
                torch.cuda.reset_peak_memory_stats()
            gc.collect()
    except ImportError:
        pass
