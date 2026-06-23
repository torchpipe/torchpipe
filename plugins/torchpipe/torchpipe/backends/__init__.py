"""
Python backends for TorchPipe.

This module provides Python implementations of various backends,
including TensorRT inference and CUDA stream synchronization.
"""

from .base import BackendProtocol, BackendMeta, register_backend, backend
from .py_sync import PySyncTensor

__all__ = [
    "BackendProtocol",
    "BackendMeta",
    "register_backend",
    "backend",
    "PyTensorrtInferTensor",
    "PyTensorrtEngine",
    "PySyncTensor",
]


def __getattr__(name):
    if name in {"PyTensorrtInferTensor", "PyTensorrtEngine"}:
        from .py_tensorrt import PyTensorrtEngine, PyTensorrtInferTensor

        exports = {
            "PyTensorrtInferTensor": PyTensorrtInferTensor,
            "PyTensorrtEngine": PyTensorrtEngine,
        }
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
