"""
Python implementation of CUDA stream synchronization backend.

This module provides PySyncTensor for managing CUDA streams in
independent thread mode, replacing the C++ SyncTensor implementation.
"""

from __future__ import annotations

import logging
from typing import Optional, List, Any, Dict

logger = logging.getLogger(__name__)

_torch_available = False
try:
    import torch
    _torch_available = True
except ImportError:
    logger.error("PyTorch is required for PySyncTensor")

try:
    import tvm_ffi
    _tvm_ffi_available = True
except ImportError:
    _tvm_ffi_available = False
    logger.debug("tvm_ffi not available, some features may be limited")

from .base import BackendBase, BackendMeta, register_backend
from .cuda_utils import (
    CUDAStreamManager,
    CUDAStreamError,
    torch_not_use_default_stream,
    get_current_device,
)

TASK_INDEX_KEY = "instance_index"


class PySyncTensor(BackendBase):
    """
    CUDA stream synchronization backend.
    
    This backend manages CUDA streams in independent thread mode,
    ensuring proper synchronization between the default stream and
    the current stream.
    
    The key responsibilities are:
    1. Detect independent thread mode (TASK_INDEX_KEY presence)
    2. Create and bind a non-default CUDA stream
    3. Set TVMFFI environment stream for cross-framework compatibility
    4. Synchronize streams during forward execution
    """
    
    def __init__(self):
        """Initialize the sync backend."""
        super().__init__()
        
        self._owned_backend: Optional[Any] = None
        self._stream: Optional[torch.cuda.Stream] = None
        self._event: Optional[torch.cuda.Event] = None
        self._need_sync: bool = False
        self._independent_thread_index: int = -1
        self._original_stream: Optional[torch.cuda.Stream] = None
        self._device_id: int = 0
    
    def init(
        self,
        config: Dict[str, str],
        options: Optional[Any] = None
    ) -> None:
        """
        Initialize the sync backend.
        
        Args:
            config: Configuration parameters
                - TASK_INDEX_KEY: Required for independent thread mode
                - device: Optional device ID
            options: Optional advanced configuration
            
        Raises:
            RuntimeError: If not in independent thread mode
        """
        super().init(config, options)
        
        if not _torch_available:
            raise CUDAStreamError("PyTorch is required")
        
        if TASK_INDEX_KEY not in config:
            raise RuntimeError(
                "You are not in an independent thread mode, "
                "SyncTensor requires TASK_INDEX_KEY to be present in config"
            )
        
        try:
            self._independent_thread_index = int(config[TASK_INDEX_KEY])
        except ValueError:
            self._independent_thread_index = 0
        
        device_str = config.get("device", "0")
        try:
            self._device_id = int(device_str)
        except ValueError:
            self._device_id = 0
        
        CUDAStreamManager.synchronize_stream(
            CUDAStreamManager.get_current_stream(self._device_id)
        )
        
        self._need_sync = torch_not_use_default_stream(self._device_id, high_priority=True)
        
        if self._need_sync:
            self._stream = CUDAStreamManager.get_current_stream(self._device_id)
            self._event = CUDAStreamManager.create_event()
            
            logger.debug(
                f"PySyncTensor created non-default stream for device {self._device_id}, "
                f"thread_index={self._independent_thread_index}"
            )
        
        self._setup_tvm_ffi_env()
        
        self._setup_dlpack_allocator()
        
        logger.debug(
            f"PySyncTensor initialized: thread_index={self._independent_thread_index}, "
            f"need_sync={self._need_sync}"
        )
    
    def _setup_tvm_ffi_env(self) -> None:
        """Set up TVM FFI environment stream for cross-framework compatibility."""
        if not _tvm_ffi_available:
            return
        
        try:
            import tvm_ffi
            from tvm_ffi import TVMFFIEnvSetStream
            
            if self._stream is not None:
                stream_ptr = self._stream.cuda_stream
                original_ptr = TVMFFIEnvSetStream(
                    2,  # kDLCUDA
                    self._device_id,
                    stream_ptr,
                    None
                )
                logger.debug(f"TVMFFIEnvSetStream called for device {self._device_id}")
        except Exception as e:
            logger.warning(f"Failed to set TVMFFI environment stream: {e}")
    
    def _setup_dlpack_allocator(self) -> None:
        """Set up DLPack managed tensor allocator."""
        if not _tvm_ffi_available:
            return
        
        try:
            import tvm_ffi
            from tvm_ffi import TVMFFIEnvSetDLPackManagedTensorAllocator
            
            def torch_allocator():
                return _TorchDLPackAllocator()
            
            TVMFFIEnvSetDLPackManagedTensorAllocator(torch_allocator(), 0, None)
            logger.debug("DLPack allocator set up successfully")
        except Exception as e:
            logger.warning(f"Failed to set DLPack allocator: {e}")
    
    def forward(self, ios: List[Any]) -> None:
        """
        Execute forward with stream synchronization.
        
        Args:
            ios: List of input/output dictionaries
        """
        if self._owned_backend is not None:
            self._sync_before_forward()
        
        self._execute_forward(ios)
        
        if self._need_sync:
            self._sync_after_forward()
    
    def _sync_before_forward(self) -> None:
        """Synchronize streams before forward execution."""
        if not self._need_sync or self._event is None:
            return
        
        default_stream = CUDAStreamManager.get_default_stream(self._device_id)
        current_stream = CUDAStreamManager.get_current_stream(self._device_id)
        
        self._event.record(default_stream)
        current_stream.wait_event(self._event)
    
    def _execute_forward(self, ios: List[Any]) -> None:
        """Execute the actual forward operation."""
        if self._owned_backend is not None:
            if hasattr(self._owned_backend, 'forward'):
                self._owned_backend.forward(ios)
        elif hasattr(self, '_dep') and self._dep is not None:
            self._dep.forward(ios)
    
    def _sync_after_forward(self) -> None:
        """Synchronize stream after forward execution."""
        if not self._need_sync:
            return
        
        current_stream = CUDAStreamManager.get_current_stream(self._device_id)
        current_stream.synchronize()
    
    def set_owned_backend(self, backend: Any) -> None:
        """
        Set the owned backend.
        
        Args:
            backend: Backend instance to own
        """
        self._owned_backend = backend
    
    @property
    def need_sync(self) -> bool:
        """Whether stream synchronization is needed."""
        return self._need_sync
    
    @property
    def stream(self) -> Optional['torch.cuda.Stream']:
        """Get the managed CUDA stream."""
        return self._stream


class _TorchDLPackAllocator:
    """
    DLPack allocator using PyTorch memory management.
    
    This class provides a compatible allocator interface for
    TVMFFI DLPack managed tensor allocation.
    """
    
    def __init__(self):
        """Initialize the allocator."""
        self._allocations: Dict[int, torch.Tensor] = {}
        self._counter = 0
    
    def allocate(self, nbytes: int) -> int:
        """
        Allocate memory.
        
        Args:
            nbytes: Number of bytes to allocate
            
        Returns:
            Pointer to allocated memory
        """
        tensor = torch.empty(nbytes, dtype=torch.int8, device='cuda')
        
        handle = self._counter
        self._counter += 1
        self._allocations[handle] = tensor
        
        return tensor.data_ptr()
    
    def free(self, ptr: int) -> None:
        """
        Free allocated memory.
        
        Args:
            ptr: Pointer to free
        """
        handles_to_remove = [
            h for h, t in self._allocations.items()
            if t.data_ptr() == ptr
        ]
        for h in handles_to_remove:
            del self._allocations[h]


class PyStreamPool(BackendBase):
    """
    Stream pool backend for concurrent execution.
    
    This backend manages a pool of CUDA streams that can be used
    for concurrent kernel execution across multiple requests.
    """
    
    def __init__(self):
        """Initialize the stream pool backend."""
        super().__init__()
        
        self._stream_pool: Optional[Any] = None
        self._num_streams: int = 4
        self._dep: Optional[Any] = None
    
    def init(
        self,
        config: Dict[str, str],
        options: Optional[Any] = None
    ) -> None:
        """
        Initialize the stream pool.
        
        Args:
            config: Configuration parameters
                - num_streams: Number of streams in the pool (default: 4)
            options: Optional advanced configuration
        """
        super().init(config, options)
        
        if not _torch_available:
            raise CUDAStreamError("PyTorch is required")
        
        try:
            self._num_streams = int(config.get("num_streams", "4"))
        except ValueError:
            self._num_streams = 4
        
        from .cuda_utils import StreamPool
        self._stream_pool = StreamPool(
            num_streams=self._num_streams,
            high_priority=True
        )
        
        logger.debug(f"PyStreamPool initialized with {self._num_streams} streams")
    
    def forward(self, ios: List[Any]) -> None:
        """
        Execute forward using a stream from the pool.
        
        Args:
            ios: List of input/output dictionaries
        """
        if self._stream_pool is None:
            raise CUDAStreamError("Stream pool not initialized")
        
        with self._stream_pool.use_stream() as (index, stream_event):
            original_stream = CUDAStreamManager.get_current_stream()
            
            stream_event.event.record(original_stream)
            
            with torch.cuda.stream(stream_event.stream):
                stream_event.stream.wait_event(stream_event.event)
                
                if self._dep is not None:
                    self._dep.forward(ios)
                
                stream_event.event.record(stream_event.stream)
            
            original_stream.wait_event(stream_event.event)
    
    def set_dependency(self, dep: Any) -> None:
        """
        Set the dependency backend.
        
        Args:
            dep: Dependency backend
        """
        self._dep = dep
    
    @property
    def num_streams(self) -> int:
        """Get the number of streams in the pool."""
        return self._num_streams


register_backend(
    "PySyncTensor",
    PySyncTensor,
    BackendMeta(
        name="PySyncTensor",
        version="1.0.0",
        description="Python implementation of CUDA stream synchronization",
        tags=["cuda", "stream", "sync"],
        requires_cuda=True,
    )
)

register_backend(
    "PyStreamPool",
    PyStreamPool,
    BackendMeta(
        name="PyStreamPool",
        version="1.0.0",
        description="Python implementation of CUDA stream pool",
        tags=["cuda", "stream", "pool"],
        requires_cuda=True,
    )
)

__all__ = [
    "PySyncTensor",
    "PyStreamPool",
]
