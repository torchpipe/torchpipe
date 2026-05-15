#!/usr/bin/env python
"""
Error handling and resource management example.

This example demonstrates:
- Proper error handling with TensorRT backend
- Resource cleanup using context managers
- Memory leak prevention
- Graceful error recovery
"""

import torch
import tempfile
import os
import gc
import atexit

from torchpipe.backends.py_tensorrt import PyTensorrtEngine, PyTensorrtInferTensor, ProfileConfig
from torchpipe.backends.trt_utils import TensorRTError, ProfileError


# Global cleanup registry
_cleanup_paths = []


def _cleanup_on_exit():
    """Cleanup function registered with atexit."""
    for path in _cleanup_paths:
        if os.path.exists(path):
            try:
                os.remove(path)
            except Exception:
                pass


atexit.register(_cleanup_on_exit)


class ModelManager:
    """
    Context manager for model lifecycle management.
    
    This class ensures proper resource cleanup even when errors occur.
    """
    
    def __init__(self, model_path: str, use_fp16: bool = True):
        self.model_path = model_path
        self.use_fp16 = use_fp16
        self.engine = None
        self.backend = None
        self._onnx_path = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
        return False
    
    def create_onnx(self, model, input_shape):
        """Create ONNX model from PyTorch model."""
        model.eval()
        self._onnx_path = tempfile.NamedTemporaryFile(suffix='.onnx', delete=False).name
        _cleanup_paths.append(self._onnx_path)
        
        dummy_input = torch.randn(input_shape)
        torch.onnx.export(
            model,
            dummy_input,
            self._onnx_path,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        )
        
        return self._onnx_path
    
    def build_engine(self, profiles=None):
        """Build TensorRT engine."""
        if self._onnx_path is None:
            raise ValueError("ONNX model not created. Call create_onnx first.")
        
        self.engine = PyTensorrtEngine(instance_num=1)
        
        if profiles is None:
            profiles = [
                ProfileConfig(
                    min_shapes={'input': (1, 3, 224, 224)},
                    opt_shapes={'input': (4, 3, 224, 224)},
                    max_shapes={'input': (8, 3, 224, 224)},
                )
            ]
        
        self.engine.load_from_onnx(
            self._onnx_path,
            profiles=profiles,
            fp16_mode=self.use_fp16
        )
        
        return self.engine
    
    def create_backend(self):
        """Create inference backend."""
        if self.engine is None:
            raise ValueError("Engine not built. Call build_engine first.")
        
        self.backend = PyTensorrtInferTensor()
        self.backend._engine = self.engine
        self.backend._context = self.engine.get_or_create_context(0)
        self.backend._io_info = self.engine.get_io_info(0)
        self.backend._initialized = True
        self.backend._input_finish_event = torch.cuda.Event()
        
        return self.backend
    
    def infer(self, input_tensor):
        """
        Run inference with error handling.
        
        Args:
            input_tensor: Input tensor (will be moved to CUDA if needed)
            
        Returns:
            Output tensor
            
        Raises:
            TensorRTError: If inference fails
        """
        if self.backend is None:
            raise ValueError("Backend not created. Call create_backend first.")
        
        # Ensure input is on CUDA
        if not input_tensor.is_cuda:
            input_tensor = input_tensor.cuda()
        
        # Ensure input is contiguous
        if not input_tensor.is_contiguous():
            input_tensor = input_tensor.contiguous()
        
        try:
            io_dict = {"data": input_tensor}
            self.backend.forward([io_dict])
            return io_dict["result"]
        except TensorRTError:
            raise
        except Exception as e:
            raise TensorRTError(f"Inference failed: {e}", cause=e)
    
    def cleanup(self):
        """Release all resources."""
        if self.backend is not None:
            if hasattr(self.backend, 'release'):
                self.backend.release()
            self.backend = None
        
        if self.engine is not None:
            if hasattr(self.engine, 'release'):
                self.engine.release()
            self.engine = None
        
        # Force garbage collection
        gc.collect()
        
        # Synchronize CUDA
        if torch.cuda.is_available():
            torch.cuda.synchronize()


def demonstrate_error_handling():
    """Demonstrate error handling patterns."""
    print("=" * 60)
    print("Error Handling Demonstration")
    print("=" * 60)
    
    # Create a simple model
    class SimpleModel(torch.nn.Module):
        def forward(self, x):
            return x * 2
    
    model = SimpleModel()
    
    # Pattern 1: Using context manager
    print("\n1. Using context manager for automatic cleanup:")
    with ModelManager("simple_model", use_fp16=False) as manager:
        manager.create_onnx(model, (1, 3, 224, 224))
        manager.build_engine()
        manager.create_backend()
        
        input_tensor = torch.randn((4, 3, 224, 224))
        result = manager.infer(input_tensor)
        print(f"   Input shape: {input_tensor.shape}")
        print(f"   Output shape: {result.shape}")
    
    print("   Resources automatically cleaned up!")
    
    # Pattern 2: Try-finally cleanup
    print("\n2. Using try-finally for manual cleanup:")
    manager = ModelManager("simple_model", use_fp16=False)
    try:
        manager.create_onnx(model, (1, 3, 224, 224))
        manager.build_engine()
        manager.create_backend()
        
        input_tensor = torch.randn((4, 3, 224, 224))
        result = manager.infer(input_tensor)
        print(f"   Input shape: {input_tensor.shape}")
        print(f"   Output shape: {result.shape}")
    finally:
        manager.cleanup()
        print("   Resources manually cleaned up!")
    
    # Pattern 3: Error recovery
    print("\n3. Error recovery demonstration:")
    manager = ModelManager("simple_model", use_fp16=False)
    try:
        manager.create_onnx(model, (1, 3, 224, 224))
        manager.build_engine()
        manager.create_backend()
        
        # Try invalid input
        try:
            manager.infer(None)
        except (TypeError, AttributeError, TensorRTError) as e:
            print(f"   Caught expected error: {type(e).__name__}")
            print("   Recovering...")
        
        # Continue with valid input
        input_tensor = torch.randn((4, 3, 224, 224))
        result = manager.infer(input_tensor)
        print(f"   Recovery successful! Output shape: {result.shape}")
    finally:
        manager.cleanup()


def demonstrate_memory_management():
    """Demonstrate memory management patterns."""
    print("\n" + "=" * 60)
    print("Memory Management Demonstration")
    print("=" * 60)
    
    class SimpleModel(torch.nn.Module):
        def forward(self, x):
            return x * 2
    
    model = SimpleModel()
    
    # Monitor memory usage
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    
    print("\n1. Running multiple inferences with proper cleanup:")
    
    with ModelManager("simple_model", use_fp16=False) as manager:
        manager.create_onnx(model, (1, 3, 224, 224))
        manager.build_engine()
        manager.create_backend()
        
        for i in range(10):
            input_tensor = torch.randn((4, 3, 224, 224))
            result = manager.infer(input_tensor)
            
            # Explicitly delete tensors
            del input_tensor
            del result
        
        # Force garbage collection
        gc.collect()
    
    if torch.cuda.is_available():
        peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024
        print(f"   Peak memory usage: {peak_memory:.2f} MB")
    
    print("\n2. Memory leak prevention:")
    
    # Bad pattern: Not cleaning up
    print("   Bad pattern (not cleaning up):")
    for _ in range(5):
        manager = ModelManager("simple_model", use_fp16=False)
        manager.create_onnx(model, (1, 3, 224, 224))
        manager.build_engine()
        manager.create_backend()
        # Note: cleanup() is not called!
        # This is a bad pattern!
    
    # Good pattern: Always cleanup
    print("   Good pattern (always cleaning up):")
    for _ in range(5):
        manager = ModelManager("simple_model", use_fp16=False)
        try:
            manager.create_onnx(model, (1, 3, 224, 224))
            manager.build_engine()
            manager.create_backend()
        finally:
            manager.cleanup()


def main():
    """Main function."""
    print("=" * 60)
    print("Error Handling and Resource Management Example")
    print("=" * 60)
    
    demonstrate_error_handling()
    demonstrate_memory_management()
    
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print("""
Key takeaways:
1. Always use context managers or try-finally for resource cleanup
2. Release resources explicitly when done
3. Handle errors gracefully and recover when possible
4. Monitor memory usage to detect leaks
5. Force garbage collection after heavy operations
    """)
    
    print("✓ Example complete!")


if __name__ == "__main__":
    main()
