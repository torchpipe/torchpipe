"""
CUDA stream management utilities.

This module provides utilities for managing CUDA streams and events,
which are essential for asynchronous GPU computation and stream synchronization.
"""

from __future__ import annotations

import logging
from collections import deque
from contextlib import contextmanager
from typing import Optional, List, Tuple, Deque
from dataclasses import dataclass

logger = logging.getLogger(__name__)

_torch_available = False
try:
    import torch
    _torch_available = True
except ImportError:
    logger.warning("PyTorch not available, CUDA utilities will be limited")


class CUDAError(Exception):
    """Exception for CUDA-related errors."""
    pass


class CUDAStreamError(CUDAError):
    """Exception for CUDA stream-related errors."""
    pass


@dataclass
class StreamWithEvent:
    """Container for a CUDA stream with its associated event."""
    stream: 'torch.cuda.Stream'
    event: 'torch.cuda.Event'


class CUDAStreamManager:
    """
    CUDA stream management utilities.
    
    This class provides static methods for managing CUDA streams,
    including creation, synchronization, and event handling.
    """
    
    @staticmethod
    def is_available() -> bool:
        """Check if CUDA is available."""
        if not _torch_available:
            return False
        return torch.cuda.is_available()
    
    @staticmethod
    def get_current_stream(device: Optional[int] = None) -> 'torch.cuda.Stream':
        """
        Get the current CUDA stream.
        
        Args:
            device: Device index. If None, uses current device.
            
        Returns:
            Current CUDA stream
        """
        if not _torch_available:
            raise CUDAStreamError("PyTorch not available")
        return torch.cuda.current_stream(device)
    
    @staticmethod
    def get_default_stream(device: Optional[int] = None) -> 'torch.cuda.Stream':
        """
        Get the default CUDA stream.
        
        Args:
            device: Device index. If None, uses current device.
            
        Returns:
            Default CUDA stream
        """
        if not _torch_available:
            raise CUDAStreamError("PyTorch not available")
        return torch.cuda.default_stream(device)
    
    @staticmethod
    def create_stream(
        device: Optional[int] = None,
        priority: int = 0,
        high_priority: bool = False
    ) -> 'torch.cuda.Stream':
        """
        Create a new CUDA stream.
        
        Args:
            device: Device index. If None, uses current device.
            priority: Stream priority (0 = default, negative = higher priority)
            high_priority: If True, creates a high priority stream
            
        Returns:
            New CUDA stream
        """
        if not _torch_available:
            raise CUDAStreamError("PyTorch not available")
        
        if high_priority:
            return torch.cuda.Stream(device=device, priority=-1)
        return torch.cuda.Stream(device=device, priority=priority)
    
    @staticmethod
    def get_stream_from_pool(
        high_priority: bool = False,
        device: Optional[int] = None
    ) -> 'torch.cuda.Stream':
        """
        Get a stream from PyTorch's stream pool.
        
        This is more efficient than creating new streams repeatedly.
        
        Args:
            high_priority: If True, gets a high priority stream
            device: Device index. If None, uses current device.
            
        Returns:
            CUDA stream from the pool
        """
        if not _torch_available:
            raise CUDAStreamError("PyTorch not available")
        
        if hasattr(torch.cuda, 'get_stream_from_pool'):
            return torch.cuda.get_stream_from_pool(high_priority, device)
        else:
            priority = -1 if high_priority else 0
            return torch.cuda.Stream(device=device, priority=priority)
    
    @staticmethod
    def create_event(enable_timing: bool = False, blocking: bool = False) -> 'torch.cuda.Event':
        """
        Create a CUDA event.
        
        Args:
            enable_timing: If True, event can be used for timing
            blocking: If True, event will block when synchronized
            
        Returns:
            New CUDA event
        """
        if not _torch_available:
            raise CUDAStreamError("PyTorch not available")
        
        flags = 0
        if enable_timing:
            flags |= 0x01
        if blocking:
            flags |= 0x02
        
        return torch.cuda.Event(flags)
    
    @staticmethod
    def record_event(
        stream: 'torch.cuda.Stream',
        event: Optional['torch.cuda.Event'] = None
    ) -> 'torch.cuda.Event':
        """
        Record an event on a stream.
        
        Args:
            stream: CUDA stream to record on
            event: Existing event to use, or None to create new one
            
        Returns:
            The recorded event
        """
        if event is None:
            event = CUDAStreamManager.create_event()
        event.record(stream)
        return event
    
    @staticmethod
    def wait_event(
        stream: 'torch.cuda.Stream',
        event: 'torch.cuda.Event'
    ) -> None:
        """
        Make a stream wait for an event.
        
        Args:
            stream: CUDA stream that will wait
            event: CUDA event to wait for
        """
        stream.wait_event(event)
    
    @staticmethod
    def synchronize_stream(stream: 'torch.cuda.Stream') -> None:
        """
        Synchronize a CUDA stream.
        
        Args:
            stream: CUDA stream to synchronize
        """
        stream.synchronize()
    
    @staticmethod
    def synchronize_device(device: Optional[int] = None) -> None:
        """
        Synchronize all operations on a device.
        
        Args:
            device: Device index. If None, uses current device.
        """
        if not _torch_available:
            raise CUDAStreamError("PyTorch not available")
        torch.cuda.synchronize(device)
    
    @staticmethod
    def is_using_default_stream(device: Optional[int] = None) -> bool:
        """
        Check if the current stream is the default stream.
        
        Args:
            device: Device index. If None, uses current device.
            
        Returns:
            True if current stream is the default stream
        """
        if not _torch_available:
            raise CUDAStreamError("PyTorch not available")
        return torch.cuda.current_stream(device) == torch.cuda.default_stream(device)
    
    @staticmethod
    def sync_streams(
        source_stream: 'torch.cuda.Stream',
        target_stream: 'torch.cuda.Stream'
    ) -> 'torch.cuda.Event':
        """
        Synchronize two streams using an event.
        
        Records an event on the source stream and makes the target
        stream wait for that event.
        
        Args:
            source_stream: Stream to record event on
            target_stream: Stream that will wait
            
        Returns:
            The event used for synchronization
        """
        event = CUDAStreamManager.create_event()
        event.record(source_stream)
        target_stream.wait_event(event)
        return event


@contextmanager
def cuda_stream(stream: 'torch.cuda.Stream'):
    """
    Context manager for using a specific CUDA stream.
    
    Usage:
        with cuda_stream(my_stream):
            # Operations here use my_stream
            tensor = torch.randn(10, device='cuda')
    
    Args:
        stream: CUDA stream to use
        
    Yields:
        The CUDA stream
    """
    if not _torch_available:
        raise CUDAStreamError("PyTorch not available")
    
    with torch.cuda.stream(stream):
        yield stream


class StreamPool:
    """
    Pool of CUDA streams for concurrent execution.
    
    This class manages a pool of CUDA streams that can be acquired
    and released, enabling efficient concurrent kernel execution.
    """
    
    def __init__(
        self,
        num_streams: int = 4,
        high_priority: bool = True,
        device: Optional[int] = None
    ):
        """
        Initialize the stream pool.
        
        Args:
            num_streams: Number of streams in the pool
            high_priority: If True, creates high priority streams
            device: Device index. If None, uses current device.
        """
        if not _torch_available:
            raise CUDAStreamError("PyTorch not available")
        
        if num_streams <= 0 or num_streams > 32:
            raise ValueError("num_streams must be between 1 and 32")

        self._num_streams = num_streams
        self._device = device
        self._streams: List[StreamWithEvent] = []
        # Use deque for O(1) popleft() instead of list.pop(0) which is O(n)
        self._available: Deque[int] = deque(range(num_streams))
        self._in_use: set = set()

        for _ in range(num_streams):
            stream = CUDAStreamManager.get_stream_from_pool(high_priority, device)
            event = CUDAStreamManager.create_event()
            self._streams.append(StreamWithEvent(stream, event))

    def acquire(self) -> Tuple[int, StreamWithEvent]:
        """
        Acquire a stream from the pool.

        Returns:
            Tuple of (stream_index, StreamWithEvent)

        Raises:
            CUDAStreamError: If no streams are available
        """
        if not self._available:
            raise CUDAStreamError("No streams available in pool")

        # Use popleft() for O(1) operation instead of pop(0) which is O(n)
        index = self._available.popleft()
        self._in_use.add(index)
        return index, self._streams[index]
    
    def release(self, index: int) -> None:
        """
        Release a stream back to the pool.
        
        Args:
            index: Stream index to release
            
        Raises:
            ValueError: If index is invalid or not in use
        """
        if index < 0 or index >= self._num_streams:
            raise ValueError(f"Invalid stream index: {index}")
        
        if index not in self._in_use:
            raise ValueError(f"Stream {index} is not in use")
        
        self._in_use.remove(index)
        self._available.append(index)
    
    def get_stream(self, index: int) -> StreamWithEvent:
        """
        Get a stream by index without acquiring it.
        
        Args:
            index: Stream index
            
        Returns:
            StreamWithEvent for the given index
        """
        return self._streams[index]
    
    @property
    def num_streams(self) -> int:
        """Number of streams in the pool."""
        return self._num_streams
    
    @property
    def available_count(self) -> int:
        """Number of available streams."""
        return len(self._available)
    
    @contextmanager
    def use_stream(self):
        """
        Context manager for acquiring and automatically releasing a stream.
        
        Usage:
            with stream_pool.use_stream() as (index, stream_event):
                # Use stream_event.stream for operations
                pass
            # Stream is automatically released
        
        Yields:
            Tuple of (stream_index, StreamWithEvent)
        """
        index, stream_event = self.acquire()
        try:
            yield index, stream_event
        finally:
            self.release(index)


def torch_not_use_default_stream(device: int, high_priority: bool = True) -> bool:
    """
    Check if using default stream and switch to non-default if needed.
    
    This function checks if the current stream is the default stream.
    If so, it switches to a non-default stream from the pool.
    
    Args:
        device: Device index
        high_priority: If True, creates high priority stream
        
    Returns:
        True if stream was switched, False if already using non-default
    """
    if not _torch_available:
        raise CUDAStreamError("PyTorch not available")
    
    current = torch.cuda.current_stream(device)
    default = torch.cuda.default_stream(device)
    
    if current == default:
        if hasattr(torch.cuda, 'get_stream_from_pool'):
            new_stream = torch.cuda.get_stream_from_pool(high_priority, device)
        else:
            priority = -1 if high_priority else 0
            new_stream = torch.cuda.Stream(device=device, priority=priority)
        torch.cuda.set_stream(new_stream)
        return True
    
    return False


def get_current_device() -> int:
    """Get the current CUDA device index."""
    if not _torch_available:
        raise CUDAStreamError("PyTorch not available")
    return torch.cuda.current_device()
