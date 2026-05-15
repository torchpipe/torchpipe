"""
Base classes and utilities for Python backends.

This module provides the foundation for implementing Python backends
with proper type hints, metadata support, and registration utilities.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import (
    Protocol,
    Optional,
    List,
    Dict,
    Any,
    Callable,
    TypeVar,
    runtime_checkable,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")


@runtime_checkable
class BackendProtocol(Protocol):
    """
    Protocol definition for backend implementations.
    
    All backends must implement this protocol to be compatible
    with the TorchPipe scheduling system.
    """
    
    def init(
        self,
        config: Dict[str, str],
        options: Optional[Any] = None
    ) -> None:
        """
        Initialize the backend with configuration.
        
        Args:
            config: Configuration parameters as string key-value pairs
            options: Optional dictionary for advanced configuration
        """
        ...
    
    def forward(self, ios: List[Any]) -> None:
        """
        Execute the forward pass.
        
        Args:
            ios: List of input/output dictionaries. Each dictionary
                 contains input data and will be populated with results.
        """
        ...
    
    def max(self) -> int:
        """Return the maximum batch size supported."""
        ...
    
    def min(self) -> int:
        """Return the minimum batch size supported."""
        ...


@dataclass
class BackendMeta:
    """
    Metadata for backend registration.
    
    Attributes:
        name: Backend name
        version: Backend version string
        description: Brief description of the backend
        author: Author information
        tags: List of tags for categorization
        requires_cuda: Whether the backend requires CUDA
        supports_dynamic_shape: Whether dynamic shapes are supported
        supports_multi_instance: Whether multiple instances are supported
    """
    name: str
    version: str = "1.0.0"
    description: str = ""
    author: str = ""
    tags: List[str] = field(default_factory=list)
    requires_cuda: bool = False
    supports_dynamic_shape: bool = False
    supports_multi_instance: bool = False


_backend_registry: Dict[str, type] = {}
_backend_metadata: Dict[str, BackendMeta] = {}


def register_backend(
    name: str,
    backend_class: Optional[type] = None,
    meta: Optional[BackendMeta] = None,
    *,
    lazy: bool = False,
    singleton: bool = False
) -> Callable[[type], type]:
    """
    Register a Python backend.
    
    Can be used as a decorator or as a function call:
    
        @register_backend("MyBackend")
        class MyBackend:
            ...
    
    Or:
    
        register_backend("MyBackend", MyBackendClass)
    
    Args:
        name: Backend registration name
        backend_class: Backend class (optional when used as decorator)
        meta: Backend metadata
        lazy: Whether to delay initialization
        singleton: Whether to use singleton pattern
        
    Returns:
        Decorator function or the registered class
    """
    def decorator(cls: type) -> type:
        _backend_registry[name] = cls
        if meta:
            _backend_metadata[name] = meta
        else:
            _backend_metadata[name] = BackendMeta(name=name)
        
        try:
            import omniback
            
            omniback.register(name, cls)
            logger.debug(f"Registered backend '{name}' with class '{cls.__name__}'")
        except ImportError:
            logger.warning("omniback not available, backend registered locally only")
        except Exception as e:
            logger.error(f"Failed to register backend '{name}': {e}")
            raise
        
        return cls
    
    if backend_class is not None:
        return decorator(backend_class)
    
    return decorator


def backend(name: str, meta: Optional[BackendMeta] = None):
    """
    Decorator for backend registration.
    
    Usage:
        @backend("MyBackend", BackendMeta(name="MyBackend", version="1.0"))
        class MyBackend:
            def init(self, config, options=None): ...
            def forward(self, ios): ...
            def max(self): ...
            def min(self): ...
    
    Args:
        name: Backend registration name
        meta: Backend metadata
        
    Returns:
        Decorator function
    """
    return register_backend(name, meta=meta)


def get_backend(name: str) -> Optional[type]:
    """
    Get a registered backend class by name.
    
    Args:
        name: Backend name
        
    Returns:
        Backend class or None if not found
    """
    return _backend_registry.get(name)


def get_backend_meta(name: str) -> Optional[BackendMeta]:
    """
    Get backend metadata by name.
    
    Args:
        name: Backend name
        
    Returns:
        Backend metadata or None if not found
    """
    return _backend_metadata.get(name)


def list_backends() -> List[str]:
    """
    List all registered backend names.
    
    Returns:
        List of backend names
    """
    return list(_backend_registry.keys())


class BackendBase:
    """
    Base class for backend implementations.
    
    Provides default implementations for common methods.
    """
    
    def __init__(self):
        self._initialized = False
        self._config: Dict[str, str] = {}
        self._options: Optional[Any] = None
    
    def init(
        self,
        config: Dict[str, str],
        options: Optional[Any] = None
    ) -> None:
        """
        Initialize the backend.
        
        Args:
            config: Configuration parameters
            options: Optional advanced configuration
        """
        self._config = config
        self._options = options
        self._initialized = True
    
    def forward(self, ios: List[Any]) -> None:
        """
        Execute forward pass. Must be implemented by subclasses.
        
        Args:
            ios: List of input/output dictionaries
        """
        raise NotImplementedError("Subclasses must implement forward()")
    
    def max(self) -> int:
        """Return maximum batch size. Default is 1."""
        return 1
    
    def min(self) -> int:
        """Return minimum batch size. Default is 1."""
        return 1
    
    def is_initialized(self) -> bool:
        """Check if backend is initialized."""
        return self._initialized
