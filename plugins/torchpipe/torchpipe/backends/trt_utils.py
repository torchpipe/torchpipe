"""
TensorRT utility functions.

This module provides utilities for working with TensorRT engines,
including loading, saving, building from ONNX, and extracting IO information.
"""

from __future__ import annotations

import logging
import os
import tempfile
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import (
    Optional,
    List,
    Dict,
    Tuple,
    Any,
    Union,
    Sequence,
)

logger = logging.getLogger(__name__)

_torch_available = False
_torch_import_error: Optional[Exception] = None
try:
    import torch
    _torch_available = True
except ImportError as e:
    _torch_import_error = e

_tensorrt_available = False
_tensorrt_import_error: Optional[Exception] = None
try:
    import tensorrt as trt
    _tensorrt_available = True
except (ImportError, OSError) as e:
    _tensorrt_import_error = e


def _raise_torch_unavailable(feature: str) -> None:
    raise TensorRTError(
        f"{feature} requires PyTorch, but importing `torch` failed.",
        cause=_torch_import_error,
    )


def _raise_tensorrt_unavailable(feature: str) -> None:
    raise TensorRTError(
        f"{feature} requires the TensorRT Python package `tensorrt`, but import failed.",
        cause=_tensorrt_import_error,
    )


class TensorRTError(Exception):
    """
    Exception for TensorRT-related errors.
    
    Attributes:
        message: Error message
        context: Additional context information (e.g., tensor name, shape)
        cause: Original exception that caused this error
    """
    
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None, cause: Optional[Exception] = None):
        """
        Initialize TensorRTError.
        
        Args:
            message: Error message
            context: Additional context information
            cause: Original exception that caused this error
        """
        self.message = message
        self.context = context or {}
        self.cause = cause
        super().__init__(self._format_message())
    
    def _format_message(self) -> str:
        """Format the error message with context."""
        parts = [self.message]
        
        if self.context:
            context_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            parts.append(f" (context: {context_str})")
        
        if self.cause:
            parts.append(f" (caused by: {type(self.cause).__name__}: {self.cause})")
        
        return "".join(parts)
    
    def __str__(self) -> str:
        return self._format_message()


class ProfileError(TensorRTError):
    """Exception for Profile-related errors."""
    pass


class EngineError(TensorRTError):
    """Exception for Engine-related errors."""
    pass


class ContextError(TensorRTError):
    """Exception for Context-related errors."""
    pass


class InferenceError(TensorRTError):
    """Exception for Inference-related errors."""
    pass


class DataType(IntEnum):
    """TensorRT data types compatible with NetIOInfo."""
    RESERVED_INT = 0
    INT4 = 1
    INT8 = 2
    UINT8 = 3
    INT32 = 4
    INT64 = 5
    BOOL = 6
    RESERVED_FP = 32
    FP4 = 33
    FP8 = 34
    FP32 = 35
    FP16 = 36
    RESERVED_BF = 48
    BF16 = 49
    BF32 = 50
    UNKNOWN = 255


@dataclass
class Dims64:
    """Dimensions with up to 8 axes."""
    MAX_DIMS: int = 8
    nbDims: int = 0
    d: Tuple[int, ...] = (0,) * 8
    
    @classmethod
    def from_trt(cls, trt_dims) -> 'Dims64':
        """Create Dims64 from TensorRT Dims."""
        dims = cls()
        try:
            dims.nbDims = len(trt_dims)
            dims.d = tuple(trt_dims)
        except (TypeError, ValueError):
            dims.nbDims = 0
            dims.d = (0,) * cls.MAX_DIMS
        return dims
    
    @classmethod
    def from_tuple(cls, shape: Tuple[int, ...]) -> 'Dims64':
        """Create Dims64 from a tuple."""
        dims = cls()
        dims.nbDims = len(shape)
        dims.d = shape + (0,) * (cls.MAX_DIMS - len(shape))
        return dims
    
    def to_tuple(self) -> Tuple[int, ...]:
        """Convert to tuple."""
        return self.d[:self.nbDims]
    
    def __iter__(self):
        return iter(self.d[:self.nbDims])
    
    def __len__(self):
        return self.nbDims
    
    def __getitem__(self, index):
        return self.d[index]


@dataclass
class NetIOInfo:
    """
    Network input/output information.
    
    Compatible with the C++ NetIOInfo structure.
    """
    class Device(IntEnum):
        CPU = 0
        GPU = 1
    
    min: Dims64 = field(default_factory=Dims64)
    max: Dims64 = field(default_factory=Dims64)
    dtype: DataType = DataType.FP32
    device: Device = Device.GPU
    name: Optional[str] = None


NetIOInfos = Tuple[List[NetIOInfo], List[NetIOInfo]]


TRT_DTYPE_TO_DATATYPE = {}

if _tensorrt_available:
    TRT_DTYPE_TO_DATATYPE = {
        trt.DataType.FLOAT: DataType.FP32,
        trt.DataType.HALF: DataType.FP16,
        trt.DataType.INT8: DataType.INT8,
        trt.DataType.INT32: DataType.INT32,
        trt.DataType.BOOL: DataType.BOOL,
    }

    if hasattr(trt.DataType, 'INT64'):
        TRT_DTYPE_TO_DATATYPE[trt.DataType.INT64] = DataType.INT64
    if hasattr(trt.DataType, 'BF16'):
        TRT_DTYPE_TO_DATATYPE[trt.DataType.BF16] = DataType.BF16
    if hasattr(trt.DataType, 'FP8'):
        TRT_DTYPE_TO_DATATYPE[trt.DataType.FP8] = DataType.FP8
    if hasattr(trt.DataType, 'INT4'):
        TRT_DTYPE_TO_DATATYPE[trt.DataType.INT4] = DataType.INT4

DATATYPE_TO_TORCH_DTYPE = {
    DataType.FP32: torch.float32 if _torch_available else None,
    DataType.FP16: torch.float16 if _torch_available else None,
    DataType.BF16: torch.bfloat16 if _torch_available else None,
    DataType.INT8: torch.int8 if _torch_available else None,
    DataType.INT32: torch.int32 if _torch_available else None,
    DataType.INT64: torch.int64 if _torch_available else None,
    DataType.BOOL: torch.bool if _torch_available else None,
    DataType.UINT8: torch.uint8 if _torch_available else None,
}

TORCH_DTYPE_TO_DATATYPE = {v: k for k, v in DATATYPE_TO_TORCH_DTYPE.items() if v is not None}


def trt_dtype_to_datatype(trt_dtype) -> DataType:
    """Convert TensorRT DataType to our DataType enum."""
    if not _tensorrt_available:
        _raise_tensorrt_unavailable("TensorRT dtype conversion")
    return TRT_DTYPE_TO_DATATYPE.get(trt_dtype, DataType.UNKNOWN)


def datatype_to_torch_dtype(dtype: DataType) -> 'torch.dtype':
    """Convert DataType to PyTorch dtype."""
    if not _torch_available:
        _raise_torch_unavailable("Converting TensorRT dtype to torch dtype")
    torch_dtype = DATATYPE_TO_TORCH_DTYPE.get(dtype)
    if torch_dtype is None:
        raise TensorRTError(f"Unsupported DataType: {dtype}")
    return torch_dtype


def torch_dtype_to_datatype(torch_dtype: 'torch.dtype') -> DataType:
    """Convert PyTorch dtype to DataType."""
    if not _torch_available:
        _raise_torch_unavailable("Converting torch dtype to TensorRT dtype")
    return TORCH_DTYPE_TO_DATATYPE.get(torch_dtype, DataType.UNKNOWN)


# Module-level cached logger instance for better performance
_cached_trt_logger: Optional['trt.Logger'] = None


def get_trt_logger() -> 'trt.Logger':
    """
    Get or create a TensorRT logger.

    This function caches the logger instance at module level to avoid
    repeated creation overhead.

    Returns:
        TensorRT Logger instance
    """
    global _cached_trt_logger

    if not _tensorrt_available:
        _raise_tensorrt_unavailable("Creating a TensorRT logger")

    if _cached_trt_logger is None:
        _cached_trt_logger = trt.Logger(trt.Logger.WARNING)

    return _cached_trt_logger


def load_engine_from_file(
    engine_path: str,
    logger: Optional['trt.Logger'] = None
) -> 'trt.ICudaEngine':
    """
    Load a TensorRT engine from a file.
    
    Args:
        engine_path: Path to the .trt engine file
        logger: Optional TensorRT logger
        
    Returns:
        TensorRT CUDA engine
    """
    if not _tensorrt_available:
        _raise_tensorrt_unavailable("Loading a TensorRT engine")
    
    if logger is None:
        logger = get_trt_logger()
    
    if not os.path.exists(engine_path):
        raise TensorRTError(f"Engine file not found: {engine_path}")
    
    runtime = trt.Runtime(logger)
    
    with open(engine_path, 'rb') as f:
        engine_data = f.read()
    
    engine = runtime.deserialize_cuda_engine(engine_data)
    if engine is None:
        raise TensorRTError(f"Failed to deserialize engine from {engine_path}")
    
    return engine


def save_engine_to_file(
    engine: 'trt.ICudaEngine',
    engine_path: str
) -> None:
    """
    Save a TensorRT engine to a file.
    
    Args:
        engine: TensorRT CUDA engine
        engine_path: Path to save the .trt file
    """
    if not _tensorrt_available:
        _raise_tensorrt_unavailable("Saving a TensorRT engine")
    
    serialized_engine = engine.serialize()
    
    os.makedirs(os.path.dirname(engine_path) or '.', exist_ok=True)
    
    with open(engine_path, 'wb') as f:
        f.write(serialized_engine)
    
    logger.info(f"Engine saved to {engine_path}")


def onnx_to_trt(
    onnx_path: str,
    engine_path: Optional[str] = None,
    max_batch_size: int = 1,
    max_workspace_size: int = 1 << 30,
    fp16_mode: bool = True,
    int8_mode: bool = False,
    profiles: Optional[Union[List[Dict[str, Any]], List[Any]]] = None,
    logger: Optional['trt.Logger'] = None,
    **kwargs
) -> 'trt.ICudaEngine':
    """
    Convert an ONNX model to TensorRT engine.
    
    Args:
        onnx_path: Path to the ONNX model
        engine_path: Optional path to save the engine
        max_batch_size: Maximum batch size
        max_workspace_size: Maximum workspace size in bytes
        fp16_mode: Enable FP16 mode
        int8_mode: Enable INT8 mode
        profiles: List of optimization profiles for dynamic shapes
        logger: Optional TensorRT logger
        **kwargs: Additional builder flags
        
    Returns:
        TensorRT CUDA engine
    """
    if not _tensorrt_available:
        _raise_tensorrt_unavailable("Building a TensorRT engine from ONNX")
    
    if logger is None:
        logger = get_trt_logger()
    
    if not os.path.exists(onnx_path):
        raise TensorRTError(f"ONNX file not found: {onnx_path}")
    
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)
    
    with open(onnx_path, 'rb') as f:
        if not parser.parse(f.read()):
            errors = [parser.get_error(i) for i in range(parser.num_errors)]
            raise TensorRTError(f"Failed to parse ONNX: {errors}")
    
    config = builder.create_builder_config()
    
    if hasattr(config, 'set_memory_pool_limit'):
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, max_workspace_size)
    else:
        config.max_workspace_size = max_workspace_size
    
    if fp16_mode:
        config.set_flag(trt.BuilderFlag.FP16)
    
    if int8_mode:
        config.set_flag(trt.BuilderFlag.INT8)
    
    if profiles:
        for profile_data in profiles:
            profile = builder.create_optimization_profile()
            
            if hasattr(profile_data, 'min_shapes') and hasattr(profile_data, 'opt_shapes') and hasattr(profile_data, 'max_shapes'):
                # Handle ProfileConfig object
                for name in profile_data.min_shapes.keys():
                    min_shape = profile_data.min_shapes[name]
                    opt_shape = profile_data.opt_shapes[name]
                    max_shape = profile_data.max_shapes[name]
                    if min_shape and opt_shape and max_shape:
                        profile.set_shape(name, min_shape, opt_shape, max_shape)
            else:
                # Handle dictionary format
                for name, shapes in profile_data.items():
                    min_shape = shapes.get('min')
                    opt_shape = shapes.get('opt')
                    max_shape = shapes.get('max')
                    
                    if min_shape and opt_shape and max_shape:
                        profile.set_shape(name, min_shape, opt_shape, max_shape)
            
            config.add_optimization_profile(profile)
    else:
        has_dynamic_shape = False
        for i in range(network.num_inputs):
            input = network.get_input(i)
            input_shape = input.shape
            if -1 in input_shape:
                has_dynamic_shape = True
                break
        
        if has_dynamic_shape:
            profile = builder.create_optimization_profile()
            for i in range(network.num_inputs):
                input = network.get_input(i)
                input_shape = list(input.shape)
                name = input.name
                
                min_shape = []
                opt_shape = []
                max_shape = []
                
                for j, dim in enumerate(input_shape):
                    if dim == -1:
                        if j == 0:
                            min_shape.append(1)
                            opt_shape.append(max(1, max_batch_size // 2))
                            max_shape.append(max_batch_size)
                        else:
                            min_shape.append(1)
                            opt_shape.append(4)
                            max_shape.append(8)
                    else:
                        min_shape.append(dim)
                        opt_shape.append(dim)
                        max_shape.append(dim)
                
                profile.set_shape(name, tuple(min_shape), tuple(opt_shape), tuple(max_shape))
            
            config.add_optimization_profile(profile)
    
    for key, value in kwargs.items():
        if hasattr(trt.BuilderFlag, key.upper()):
            if value:
                config.set_flag(getattr(trt.BuilderFlag, key.upper()))
    
    if hasattr(builder, 'build_serialized_network'):
        serialized_engine = builder.build_serialized_network(network, config)
        if serialized_engine is None:
            raise TensorRTError("Failed to build serialized network")
        
        import numpy as np
        engine_bytes = np.ndarray(
            (serialized_engine.nbytes,), 
            dtype=np.uint8, 
            buffer=serialized_engine
        ).tobytes()
        
        runtime = trt.Runtime(logger)
        engine = runtime.deserialize_cuda_engine(engine_bytes)
    elif hasattr(builder, 'build_cuda_engine'):
        engine = builder.build_cuda_engine(network, config)
    else:
        engine = builder.build_engine(network, config)
    
    if engine is None:
        raise TensorRTError("Failed to build engine")
    
    if engine_path:
        save_engine_to_file(engine, engine_path)
    
    return engine


def get_engine_io_info(
    engine: 'trt.ICudaEngine',
    profile_index: int = 0
) -> NetIOInfos:
    """
    Get input/output information from a TensorRT engine.
    
    Args:
        engine: TensorRT CUDA engine
        profile_index: Profile index for dynamic shapes
        
    Returns:
        Tuple of (input_infos, output_infos)
    """
    if not _tensorrt_available:
        _raise_tensorrt_unavailable("Reading TensorRT engine IO info")
    
    input_infos: List[NetIOInfo] = []
    output_infos: List[NetIOInfo] = []
    
    use_new_api = hasattr(engine, 'get_tensor_name')
    has_implicit = False
    
    # Get number of I/O tensors
    if use_new_api:
        num_io_tensors = engine.num_io_tensors
    else:
        num_bindings = engine.num_bindings
        has_implicit = engine.has_implicit_batch_dimension if hasattr(engine, 'has_implicit_batch_dimension') else False
        num_io_tensors = num_bindings
    
    for i in range(num_io_tensors):
        try:
            if use_new_api:
                name = engine.get_tensor_name(i)
                dtype = engine.get_tensor_dtype(name)
                is_input = engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
                dims = engine.get_tensor_shape(name)
            else:
                name = engine.get_binding_name(i)
                dtype = engine.get_binding_dtype(i)
                is_input = engine.binding_is_input(i)
                dims = engine.get_binding_shape(i)
                has_implicit = engine.has_implicit_batch_dimension if hasattr(engine, 'has_implicit_batch_dimension') else False
            
            shape = tuple(dims)
            
            io_info = NetIOInfo()
            io_info.name = name
            io_info.dtype = trt_dtype_to_datatype(dtype)
            io_info.device = NetIOInfo.Device.GPU
            
            if -1 in shape or (not has_implicit and engine.num_optimization_profiles > 0):
                if use_new_api:
                    try:
                        profile_shapes = engine.get_tensor_profile_shape(name, profile_index)
                        if profile_shapes and len(profile_shapes) == 3:
                            min_dims, opt_dims, max_dims = profile_shapes
                            if len(min_dims) > 0 and len(max_dims) > 0:
                                io_info.min = Dims64.from_trt(min_dims)
                                io_info.max = Dims64.from_trt(max_dims)
                            else:
                                default_shape = tuple(8 if dim == -1 else dim for dim in shape)
                                io_info.min = Dims64.from_tuple(tuple(1 if dim == -1 else dim for dim in shape))
                                io_info.max = Dims64.from_tuple(default_shape)
                        else:
                            io_info.min = Dims64.from_tuple(shape)
                            io_info.max = Dims64.from_tuple(shape)
                    except Exception:
                        io_info.min = Dims64.from_tuple(tuple(1 if dim == -1 else dim for dim in shape))
                        io_info.max = Dims64.from_tuple(tuple(8 if dim == -1 else dim for dim in shape))
                else:
                    min_dims = engine.get_profile_shape(profile_index, i, trt.ProfileDimension.MIN)
                    opt_dims = engine.get_profile_shape(profile_index, i, trt.ProfileDimension.OPT)
                    max_dims = engine.get_profile_shape(profile_index, i, trt.ProfileDimension.MAX)
                    io_info.min = Dims64.from_trt(min_dims)
                    io_info.max = Dims64.from_trt(max_dims)
            else:
                io_info.min = Dims64.from_tuple(shape)
                io_info.max = Dims64.from_tuple(shape)
            
            if is_input:
                input_infos.append(io_info)
            else:
                output_infos.append(io_info)
        except Exception as e:
            logger.warning(f"Failed to get IO info for index {i}: {e}")
            continue
    
    return (input_infos, output_infos)


def get_context_io_info(
    context: 'trt.IExecutionContext',
    profile_index: int = 0
) -> NetIOInfos:
    """
    Get input/output information from a TensorRT execution context.
    
    Args:
        context: TensorRT execution context
        profile_index: Profile index for dynamic shapes
        
    Returns:
        Tuple of (input_infos, output_infos)
    """
    if not _tensorrt_available:
        _raise_tensorrt_unavailable("Reading TensorRT context IO info")
    
    engine = context.engine
    return get_engine_io_info(engine, profile_index)


def create_context(
    engine: 'trt.ICudaEngine',
    profile_index: int = 0
) -> 'trt.IExecutionContext':
    """
    Create an execution context from an engine.
    
    Args:
        engine: TensorRT CUDA engine
        profile_index: Profile index for the context
        
    Returns:
        TensorRT execution context
    """
    if not _tensorrt_available:
        _raise_tensorrt_unavailable("Creating a TensorRT execution context")
    
    context = engine.create_execution_context()
    
    if engine.num_optimization_profiles > 1:
        # Use the current CUDA stream for async operations
        if _torch_available:
            current_stream = torch.cuda.current_stream()
            context.set_optimization_profile_async(profile_index, current_stream.cuda_stream)
        else:
            # Fallback to synchronous API if torch is not available
            context.set_optimization_profile(profile_index)
    
    return context


def is_all_positive(io_infos: NetIOInfos) -> bool:
    """
    Check if all dimensions in IO info are positive.
    
    Args:
        io_infos: Network IO information
        
    Returns:
        True if all dimensions are positive
    """
    for io_list in io_infos:
        for io_info in io_list:
            for dim in io_info.min:
                if dim <= 0:
                    return False
            for dim in io_info.max:
                if dim <= 0:
                    return False
    return True


def element_size(dtype: DataType) -> int:
    """
    Get the element size in bytes for a data type.
    
    Args:
        dtype: Data type
        
    Returns:
        Size in bytes
    """
    sizes = {
        DataType.INT4: 1,
        DataType.INT8: 1,
        DataType.UINT8: 1,
        DataType.FP8: 1,
        DataType.FP16: 2,
        DataType.BF16: 2,
        DataType.INT32: 4,
        DataType.FP32: 4,
        DataType.BF32: 4,
        DataType.INT64: 8,
        DataType.BOOL: 1,
    }
    return sizes.get(dtype, 4)


def match_shape(shape1: Tuple[int, ...], shape2: Tuple[int, ...]) -> bool:
    """
    Check if two shapes match (ignoring -1 wildcard dimensions).
    
    Args:
        shape1: First shape
        shape2: Second shape
        
    Returns:
        True if shapes match
    """
    if len(shape1) != len(shape2):
        return False
    
    for d1, d2 in zip(shape1, shape2):
        if d1 != -1 and d2 != -1 and d1 != d2:
            return False
    
    return True


class TensorRTAllocator:
    """
    Memory allocator for TensorRT using PyTorch.
    
    This class provides a PyTorch-based memory allocator that can be
    used with TensorRT for efficient memory management.
    """
    
    def __init__(self, device: Optional[int] = None):
        """
        Initialize the allocator.
        
        Args:
            device: CUDA device index
        """
        if not _torch_available:
            _raise_torch_unavailable("TensorRTAllocator")
        
        self._device = device if device is not None else torch.cuda.current_device()
        self._allocations: Dict[int, torch.Tensor] = {}
        self._counter = 0
    
    def allocate(self, size: int) -> int:
        """
        Allocate memory.
        
        Args:
            size: Size in bytes
            
        Returns:
            Allocation handle
        """
        num_elements = (size + 3) // 4
        tensor = torch.empty(num_elements, dtype=torch.int8, device=f'cuda:{self._device}')
        
        handle = self._counter
        self._counter += 1
        self._allocations[handle] = tensor
        
        return handle
    
    def deallocate(self, handle: int) -> None:
        """
        Deallocate memory.
        
        Args:
            handle: Allocation handle
        """
        if handle in self._allocations:
            del self._allocations[handle]
    
    def get_pointer(self, handle: int) -> int:
        """
        Get the memory pointer for an allocation.
        
        Args:
            handle: Allocation handle
            
        Returns:
            Memory pointer
        """
        if handle not in self._allocations:
            raise TensorRTError(f"Invalid allocation handle: {handle}")
        return self._allocations[handle].data_ptr()
    
    def deallocate_all(self) -> None:
        """Deallocate all memory."""
        self._allocations.clear()
