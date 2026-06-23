"""
Python implementation of TensorRT inference backend.

This module provides PyTensorrtEngine for engine management and
PyTensorrtInferTensor for inference execution, replacing the C++
TensorrtInferTensor implementation.

Compatible with TensorRT 9.3 and TensorRT >= 10. Minimum version: TensorRT 9.3.

Features:
- Multiple optimization profiles support
- Dynamic shapes
- CUDA stream synchronization
- Zero-copy tensor operations via DLPack
- tvm_ffi integration
"""

from __future__ import annotations

import logging
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple, Union, Callable

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

_omniback_available = False
try:
    import omniback
    _omniback_available = True
except ImportError:
    pass

_tvm_ffi_available = False
try:
    import tvm_ffi
    _tvm_ffi_available = True
except ImportError:
    pass

from .base import BackendBase, BackendMeta, register_backend
from .trt_utils import (
    TensorRTError,
    ProfileError,
    EngineError,
    ContextError,
    InferenceError,
    DataType,
    Dims64,
    NetIOInfo,
    NetIOInfos,
    get_trt_logger,
    load_engine_from_file,
    save_engine_to_file,
    onnx_to_trt,
    get_engine_io_info,
    create_context,
    is_all_positive,
    element_size,
    match_shape,
    trt_dtype_to_datatype,
    datatype_to_torch_dtype,
    torch_dtype_to_datatype,
)
from .cuda_utils import (
    CUDAStreamManager,
    CUDAStreamError,
)

# Task keys
TASK_DATA_KEY = "data"
TASK_RESULT_KEY = "result"
TASK_OUTPUT_KEY = "output"
TASK_ENGINE_KEY = "engine"
TASK_IO_INFO_KEY = "net_io_infos"
TASK_INDEX_KEY = "instance_index"


def _raise_py_tensorrt_unavailable() -> None:
    raise TensorRTError(
        "PyTensorrtTensor requires the TensorRT Python package `tensorrt`, "
        "but import failed. This does not affect the C++ TensorRT backend or "
        "`python -m torchpipe.utils.encrypt`.",
        cause=_tensorrt_import_error,
    )


@dataclass
class ProfileConfig:
    """
    Configuration for an optimization profile.
    
    Attributes:
        min_shapes: Minimum shapes for each input tensor
        opt_shapes: Optimal shapes for each input tensor  
        max_shapes: Maximum shapes for each input tensor
    """
    min_shapes: Dict[str, Tuple[int, ...]]
    opt_shapes: Dict[str, Tuple[int, ...]]
    max_shapes: Dict[str, Tuple[int, ...]]
    
    def to_dict(self) -> Dict[str, Dict[str, Tuple[int, ...]]]:
        """Convert to dictionary format for trt_utils."""
        result = {}
        for name in self.min_shapes.keys():
            result[name] = {
                'min': self.min_shapes[name],
                'opt': self.opt_shapes[name],
                'max': self.max_shapes[name]
            }
        return result


@dataclass
class PyTensorrtConfig:
    """
    Configuration for PyTensorrtInferTensor.
    
    This provides a structured way to configure the backend
    with type safety and validation.
    """
    model: str
    model_type: str = "auto"  # auto, onnx, trt
    instance_num: int = 1
    instance_index: int = 0
    model_cache: Optional[str] = None
    fp16_mode: bool = True
    int8_mode: bool = False
    max_workspace_size: int = 1 << 30
    profiles: Optional[List[ProfileConfig]] = None
    
    @classmethod
    def from_dict(cls, config: Dict[str, str]) -> 'PyTensorrtConfig':
        """Create config from dictionary."""
        return cls(
            model=config.get("model", ""),
            model_type=config.get("model_type", "auto"),
            instance_num=int(config.get("instance_num", "1")),
            instance_index=int(config.get("instance_index", "0")),
            model_cache=config.get("model_cache") or config.get("model::cache"),
        )


class PyTensorrtEngine:
    """
    TensorRT engine management class.
    
    This class handles loading, saving, and managing TensorRT engines,
    including support for multiple optimization profiles.
    
    Thread-safety: This class is thread-safe for context creation.
    """
    
    def __init__(
        self,
        engine_path: Optional[str] = None,
        instance_num: int = 1
    ):
        """
        Initialize the engine manager.
        
        Args:
            engine_path: Path to the engine file (.trt or .onnx)
            instance_num: Number of instances/profiles
        """
        self._engine: Optional[trt.ICudaEngine] = None
        self._runtime: Optional[trt.Runtime] = None
        self._logger: Optional[trt.Logger] = None
        self._contexts: List[Optional[trt.IExecutionContext]] = []
        self._io_info: Optional[NetIOInfos] = None
        self._instance_num = instance_num
        self._engine_path = engine_path
        self._lock = threading.Lock()
        self._device_memory: Dict[int, Any] = {}
        self._mem_size: int = 0
    
    @property
    def engine(self) -> Optional[trt.ICudaEngine]:
        """Get the TensorRT engine."""
        return self._engine
    
    @property
    def io_info(self) -> Optional[NetIOInfos]:
        """Get the IO information."""
        return self._io_info
    
    @property
    def num_profiles(self) -> int:
        """Get the number of optimization profiles."""
        if self._engine is None:
            return 0
        return self._engine.num_optimization_profiles
    
    @property
    def instance_num(self) -> int:
        """Get the number of instances."""
        return self._instance_num
    
    @property
    def num_bindings(self) -> int:
        """Get the number of bindings."""
        if self._engine is None:
            return 0
        return self._engine.num_bindings
    
    def load_from_trt(self, path: str, logger: Optional[trt.Logger] = None) -> None:
        """
        Load engine from a .trt file.
        
        Args:
            path: Path to the .trt file
            logger: Optional TensorRT logger
        """
        if not _tensorrt_available:
            _raise_py_tensorrt_unavailable()
        
        if logger is None:
            logger = get_trt_logger()
        
        self._logger = logger
        self._engine = load_engine_from_file(path, logger)
        self._io_info = get_engine_io_info(self._engine)
        
        logger.log(trt.Logger.INFO, f"Loaded engine from {path}")
        logger.log(trt.Logger.INFO, f"Number of optimization profiles: {self.num_profiles}")
    
    def load_from_onnx(
        self,
        path: str,
        cache_path: Optional[str] = None,
        profiles: Optional[List[ProfileConfig]] = None,
        fp16_mode: bool = True,
        int8_mode: bool = False,
        max_workspace_size: int = 1 << 30,
        logger: Optional[trt.Logger] = None,
        **kwargs
    ) -> None:
        """
        Build engine from an ONNX file.
        
        Args:
            path: Path to the ONNX file
            cache_path: Optional path to cache the engine
            profiles: List of optimization profiles
            fp16_mode: Enable FP16 mode
            int8_mode: Enable INT8 mode
            max_workspace_size: Maximum workspace size
            logger: Optional TensorRT logger
            **kwargs: Additional build options
        """
        if not _tensorrt_available:
            _raise_py_tensorrt_unavailable()
        
        if logger is None:
            logger = get_trt_logger()
        
        self._logger = logger
        
        # Check cache first
        if cache_path and os.path.exists(cache_path):
            logger.log(trt.Logger.INFO, f"Loading cached engine from {cache_path}")
            self.load_from_trt(cache_path, logger)
            return
        
        # Convert ProfileConfig objects to dictionaries
        profile_list = []
        if profiles:
            for pc in profiles:
                profile_list.append(pc.to_dict())
        
        # Build engine
        self._engine = onnx_to_trt(
            onnx_path=path,
            engine_path=cache_path,
            fp16_mode=fp16_mode,
            int8_mode=int8_mode,
            max_workspace_size=max_workspace_size,
            profiles=profile_list if profile_list else None,
            logger=logger,
            **kwargs
        )
        
        self._io_info = get_engine_io_info(self._engine)
        
        logger.log(trt.Logger.INFO, f"Built engine from {path}")
        logger.log(trt.Logger.INFO, f"Number of optimization profiles: {self.num_profiles}")
    
    def create_context(self, profile_index: int = 0) -> trt.IExecutionContext:
        """
        Create an execution context.
        
        Args:
            profile_index: Profile index for the context
            
        Returns:
            TensorRT execution context
        """
        if self._engine is None:
            raise TensorRTError("Engine not loaded")
        
        if profile_index < 0 or profile_index >= max(1, self.num_profiles):
            raise ProfileError(f"Invalid profile index: {profile_index}")
        
        # Note: This method should NOT use self._lock because it's called from get_or_create_context
        # which already holds the lock. This prevents deadlock.
        return create_context(self._engine, profile_index)
    
    def get_or_create_context(self, profile_index: int = 0) -> trt.IExecutionContext:
        """
        Get or create an execution context.
        
        This method is thread-safe and will reuse existing contexts.
        
        Args:
            profile_index: Profile index
            
        Returns:
            TensorRT execution context
        """
        with self._lock:
            # Extend contexts list if needed
            while len(self._contexts) <= profile_index:
                self._contexts.append(None)
            
            # Create context if not exists
            if self._contexts[profile_index] is None:
                self._contexts[profile_index] = self.create_context(profile_index)
            
            return self._contexts[profile_index]
    
    def get_io_info(self, profile_index: int = 0) -> NetIOInfos:
        """
        Get IO information for a profile.
        
        Args:
            profile_index: Profile index
            
        Returns:
            Network IO information
        """
        if self._io_info is None:
            raise TensorRTError("Engine not loaded")
        return self._io_info
    
    def allocate_device_memory(self, size: int) -> torch.Tensor:
        """
        Allocate device memory for TensorRT execution.
        
        Args:
            size: Size in bytes
            
        Returns:
            PyTorch tensor wrapping the allocated memory
        """
        if not _torch_available:
            raise TensorRTError("PyTorch not available")
        
        # Align to 4 bytes
        num_elements = (size + 3) // 4
        return torch.empty(num_elements, dtype=torch.int32, device='cuda')
    
    def release(self) -> None:
        """
        Release all resources held by this engine.
        
        This method is thread-safe and can be called multiple times.
        After calling this method, the engine cannot be used for inference.
        """
        with self._lock:
            # Release all contexts
            self._contexts.clear()
            
            # Release device memory
            self._device_memory.clear()
            self._mem_size = 0
            
            # Release engine and runtime
            self._engine = None
            self._runtime = None
            self._io_info = None
            
            logger.debug("PyTensorrtEngine resources released")
    
    def __del__(self):
        """Destructor to ensure resources are released."""
        try:
            self.release()
        except Exception:
            pass
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.release()
        return False


class PyTensorrtInferTensor(BackendBase):
    """
    TensorRT inference backend implemented in Python.

    This class provides TensorRT inference capability with support for:
    - Multiple optimization profiles
    - Dynamic shapes
    - CUDA stream synchronization
    - Zero-copy tensor operations via DLPack
    - tvm_ffi integration

    Compatible with TensorRT 9.3 and TensorRT >= 10. Minimum version: TensorRT 9.3.

    Threading Model:
        Each backend instance is NOT thread-safe and should be used by a single
        thread only. For concurrent execution, use the multi-instance pattern:
        set instance_num > 1 to create multiple backend instances, each with its
        own TensorRT execution context. The framework will assign one instance
        per thread.

    Example:
        >>> backend = PyTensorrtInferTensor()
        >>> config = {
        ...     "model": "model.onnx",
        ...     "model_type": "onnx",
        ...     "instance_num": "1",
        ... }
        >>> backend.init(config)
        >>>
        >>> input_tensor = torch.randn(1, 3, 224, 224, device='cuda')
        >>> io_dict = {"data": input_tensor}
        >>> backend.forward([io_dict])
        >>> result = io_dict["result"]
    """
    
    def __init__(self):
        """Initialize the backend."""
        super().__init__()

        self._engine: Optional[PyTensorrtEngine] = None
        self._context: Optional[trt.IExecutionContext] = None
        self._instance_index: int = 0
        self._instance_num: int = 1
        self._io_info: Optional[NetIOInfos] = None
        self._input_finish_event: Optional[torch.cuda.Event] = None
        self._device_memory: Optional[torch.Tensor] = None
        self._mem_size: int = 0
        self._use_user_managed_mem: bool = False

        # Check TensorRT version for user-managed memory support
        if _tensorrt_available:
            trt_version = trt.__version__.split('.')
            major = int(trt_version[0])
            minor = int(trt_version[1]) if len(trt_version) > 1 else 0
            # TRT >= 10.2 supports user-managed memory
            self._use_user_managed_mem = (major == 10 and minor >= 2) or major >= 11
    
    def init(self, params: Dict[str, str], options: Optional[Any] = None) -> None:
        """
        Initialize the backend.
        
        Args:
            params: Configuration parameters (string key-value pairs)
                - model: Path to model file (.trt or .onnx)
                - model_type: Model type (onnx/trt/auto)
                - instance_num: Number of instances
                - instance_index: Instance index
                - model_cache: Path to engine cache (for ONNX)
            options: Optional omniback.Dict for advanced configuration
        """
        logger.debug(f"PyTensorrtInferTensor.init() called with params: {list(params.keys())}")
        
        super().init(params, options)
        
        if not _torch_available:
            raise TensorRTError("PyTorch is required")
        
        if not _tensorrt_available:
            raise TensorRTError("TensorRT is required")
        
        # Parse configuration
        config = PyTensorrtConfig.from_dict(params)
        
        self._instance_index = config.instance_index
        self._instance_num = config.instance_num
        
        logger.debug(f"instance_index={self._instance_index}, instance_num={self._instance_num}")
        
        assert self._instance_num >= 1 and self._instance_index >= 0, \
            f"Invalid instance configuration: num={self._instance_num}, index={self._instance_index}"
        
        # Check stream usage for multi-instance mode
        if self._instance_num > 1 and CUDAStreamManager.is_using_default_stream():
            logger.warning(
                "In multi-instance mode, the default stream is prohibited. "
                "Please use a dedicated CUDA stream with StreamGuard"
            )
        
        # Try to get engine from options (shared engine mode)
        if options is not None:
            engine_from_options = self._get_engine_from_options(options)
            if engine_from_options is not None:
                logger.debug("Using engine from options")
                self._engine = engine_from_options
                self._context = self._engine.get_or_create_context(self._instance_index)
                self._io_info = self._engine.get_io_info(self._instance_index)
        
        # Load/create engine if not from options
        if self._engine is None:
            model_path = config.model
            if not model_path:
                raise TensorRTError("'model' parameter is required")
            
            # Determine model type
            model_type = config.model_type
            if model_type == "auto" or not model_type:
                if model_path.endswith(".trt"):
                    model_type = "trt"
                elif model_path.endswith(".onnx"):
                    model_type = "onnx"
            
            logger.debug(f"Loading model: {model_path}, type: {model_type}")
            
            self._engine = PyTensorrtEngine(instance_num=self._instance_num)
            
            if model_type == "trt":
                logger.debug("Loading from TRT file...")
                self._engine.load_from_trt(model_path)
            elif model_type == "onnx":
                logger.debug(f"Building from ONNX, cache_path: {config.model_cache}")
                
                # Create default profiles if not provided
                profiles = config.profiles
                if profiles is None:
                    profiles = [
                        ProfileConfig(
                            min_shapes={'input': (1, 3, 224, 224)},
                            opt_shapes={'input': (4, 3, 224, 224)},
                            max_shapes={'input': (8, 3, 224, 224)},
                        )
                    ]
                
                self._engine.load_from_onnx(
                    model_path,
                    cache_path=config.model_cache,
                    profiles=profiles,
                    fp16_mode=config.fp16_mode,
                    int8_mode=config.int8_mode,
                    max_workspace_size=config.max_workspace_size,
                )
            else:
                raise TensorRTError(f"Unknown model type: {model_type}")
            
            logger.info("Creating execution context...")
            self._context = self._engine.get_or_create_context(self._instance_index)
            self._io_info = self._engine.get_io_info(self._instance_index)
        
        # Validate IO info
        if not is_all_positive(self._io_info):
            raise TensorRTError("Input shape is not positive")
        
        # Initialize device memory
        self._init_device_memory()
        
        # Create CUDA event for synchronization
        self._input_finish_event = torch.cuda.Event()
        
        logger.debug(
            f"PyTensorrtInferTensor instance_index={self._instance_index} "
            f"instance_num={self._instance_num} initialized"
        )
    
    def _get_engine_from_options(self, options: Any) -> Optional[PyTensorrtEngine]:
        """Try to get engine from options."""
        if options is None:
            return None
        
        if hasattr(options, 'get'):
            engine_ptr = options.get(TASK_ENGINE_KEY)
            if engine_ptr is not None:
                if isinstance(engine_ptr, PyTensorrtEngine):
                    return engine_ptr
        
        return None
    
    def _init_device_memory(self) -> None:
        """Initialize device memory for TensorRT execution."""
        if not self._use_user_managed_mem:
            return
        
        if self._context is None or self._engine is None:
            return
        
        # Get required memory size
        if hasattr(self._context, 'updateDeviceMemorySizeForShapes'):
            # TRT >= 10
            self._mem_size = self._context.updateDeviceMemorySizeForShapes()
        elif hasattr(self._engine.engine, 'getDeviceMemorySize'):
            # TRT < 10
            self._mem_size = self._engine.engine.getDeviceMemorySize()
        else:
            self._mem_size = 0
        
        if self._mem_size > 0:
            self._device_memory = self._engine.allocate_device_memory(self._mem_size)
            mem_ptr = self._device_memory.data_ptr()
            
            # Set device memory for context
            if hasattr(self._context, 'setDeviceMemoryV2'):
                self._context.setDeviceMemoryV2(mem_ptr, self._mem_size)
            elif hasattr(self._context, 'setDeviceMemory'):
                self._context.setDeviceMemory(mem_ptr)
    
    def _update_device_memory(self) -> None:
        """Update device memory if shapes changed."""
        if not self._use_user_managed_mem:
            return
        
        if self._context is None:
            return
        
        # Get updated memory size
        if hasattr(self._context, 'updateDeviceMemorySizeForShapes'):
            new_size = self._context.updateDeviceMemorySizeForShapes()
        else:
            return
        
        # Reallocate if size changed
        if new_size != self._mem_size:
            self._mem_size = new_size
            if self._mem_size > 0:
                self._device_memory = self._engine.allocate_device_memory(self._mem_size)
                mem_ptr = self._device_memory.data_ptr()
                
                if hasattr(self._context, 'setDeviceMemoryV2'):
                    self._context.setDeviceMemoryV2(mem_ptr, self._mem_size)
                elif hasattr(self._context, 'setDeviceMemory'):
                    self._context.setDeviceMemory(mem_ptr)
    
    def forward(self, ios: List[Any]) -> None:
        """
        Execute inference.

        Args:
            ios: List of omniback.Dict objects containing input data.
                 Each dict should have:
                 - 'data': input tensor(s) - can be torch.Tensor or DLPack-compatible
                 - 'output' (optional): pre-allocated output tensor(s)
                 After execution, 'result' key will contain the output.

        Warning:
            This method is NOT thread-safe. Each backend instance should be used
            by a single thread only. For concurrent execution, create multiple
            backend instances (one per thread) using the instance_num parameter.
        """
        if self._context is None:
            raise TensorRTError("Backend not initialized")

        if len(ios) != 1:
            raise TensorRTError("Only support one (batched) input with explicit batch")

        io_dict = ios[0]

        # Process inputs
        inputs = self._get_inputs(io_dict)

        # Set input shapes and tensors
        self._set_input_shapes(inputs)
        self._set_input_tensors(inputs)

        # Prepare outputs
        outputs = self._prepare_outputs(io_dict)
        self._set_output_tensors(outputs)

        # Update device memory if needed
        if self._mem_size == 0:
            self._update_device_memory()

        # Execute inference
        self._execute()

        # Set results
        self._set_result(io_dict, outputs)

        logger.debug("Forward completed successfully")
    
    def _get_inputs(self, io_dict: Any) -> List[torch.Tensor]:
        """
        Get input tensors from the IO dictionary.
        
        Supports:
        - torch.Tensor (direct)
        - DLPack-compatible objects (via __dlpack__)
        - tvm_ffi tensors
        """
        data = io_dict.get(TASK_DATA_KEY)
        
        if data is None:
            raise TensorRTError(f"'{TASK_DATA_KEY}' not found in input")
        
        # Handle single tensor or list of tensors
        if isinstance(data, (list, tuple)):
            inputs = list(data)
        else:
            inputs = [data]
        
        # Convert to torch.Tensor if needed and ensure CUDA
        for i, inp in enumerate(inputs):
            if isinstance(inp, torch.Tensor):
                # Ensure tensor is on CUDA
                if not inp.is_cuda:
                    inp = inp.cuda()
                inputs[i] = inp
                continue
            
            # Try DLPack conversion
            if hasattr(inp, '__dlpack__'):
                try:
                    tensor = torch.from_dlpack(inp)
                    if not tensor.is_cuda:
                        tensor = tensor.cuda()
                    inputs[i] = tensor
                    continue
                except Exception as e:
                    logger.warning(f"Failed to convert input {i} from DLPack: {e}")
            
            # Try tvm_ffi conversion
            if _tvm_ffi_available and hasattr(inp, 'to_dlpack'):
                try:
                    tensor = torch.from_dlpack(inp.to_dlpack())
                    if not tensor.is_cuda:
                        tensor = tensor.cuda()
                    inputs[i] = tensor
                    continue
                except Exception as e:
                    logger.warning(f"Failed to convert input {i} from tvm_ffi: {e}")
            
            raise TensorRTError(f"Input {i} is not a tensor or DLPack-compatible: {type(inp)}")
        
        return inputs
    
    def _set_input_shapes(self, inputs: List[torch.Tensor]) -> None:
        """Set input shapes in the context if they changed."""
        input_infos = self._io_info[0]
        
        for j, (inp, info) in enumerate(zip(inputs, input_infos)):
            name = info.name
            input_shape = tuple(inp.shape)
            
            # Get current shape from context
            current_shape = self._context.get_tensor_shape(name)
            
            # Update shape if:
            # 1. Current shape contains dynamic dimensions (-1)
            # 2. Shapes don't match (excluding -1 wildcards)
            needs_update = (-1 in current_shape) or (not match_shape(current_shape, input_shape))
            
            if needs_update:
                success = self._context.set_input_shape(name, input_shape)
                if not success:
                    raise TensorRTError(f"Failed to set input shape for '{name}': {input_shape}")
                self._mem_size = 0  # Trigger memory reallocation
    
    def _set_input_tensors(self, inputs: List[torch.Tensor]) -> None:
        """Set input tensor addresses in the context."""
        input_infos = self._io_info[0]
        
        for inp, info in zip(inputs, input_infos):
            name = info.name
            
            # Ensure tensor is on CUDA
            if not inp.is_cuda:
                raise TensorRTError(f"Input tensor '{name}' must be on CUDA device")
            
            # Ensure tensor is contiguous
            if not inp.is_contiguous():
                inp = inp.contiguous()
            
            # Set tensor address
            success = self._context.set_tensor_address(name, inp.data_ptr())
            if not success:
                raise TensorRTError(f"Failed to set tensor address for '{name}'")
    
    def _prepare_outputs(self, io_dict: Any) -> List[torch.Tensor]:
        """Prepare output tensors."""
        output_infos = self._io_info[1]
        outputs: List[torch.Tensor] = []
        
        # Check for user-provided outputs
        predefined_outputs = io_dict.get(TASK_OUTPUT_KEY)
        if predefined_outputs is not None:
            if isinstance(predefined_outputs, (list, tuple)):
                outputs = list(predefined_outputs)
            else:
                outputs = [predefined_outputs]
        
        predefined_size = len(outputs)
        
        # Prepare or validate outputs
        for j, info in enumerate(output_infos):
            name = info.name
            output_shape = self._context.get_tensor_shape(name)
            
            # Convert TensorRT Dims to tuple
            output_shape = tuple(output_shape)
            
            # Handle dynamic output shapes
            if -1 in output_shape:
                # For dynamic shapes, use the max shape from profile as upper bound
                max_shape = info.max.to_tuple()
                # Replace -1 with max dimension
                output_shape = tuple(
                    max_dim if dim == -1 else dim
                    for dim, max_dim in zip(output_shape, max_shape)
                )
            
            if j < predefined_size:
                # Validate user-provided output
                out = outputs[j]
                
                # Try to convert to torch.Tensor if needed
                if not isinstance(out, torch.Tensor):
                    if hasattr(out, '__dlpack__'):
                        try:
                            out = torch.from_dlpack(out)
                        except Exception:
                            out = None
                    elif _tvm_ffi_available and hasattr(out, 'to_dlpack'):
                        try:
                            out = torch.from_dlpack(out.to_dlpack())
                        except Exception:
                            out = None
                    else:
                        out = None
                
                # Check if we have a valid tensor
                if isinstance(out, torch.Tensor):
                    if not out.is_contiguous():
                        out = out.contiguous()
                    
                    # Check shape matches (allowing for dynamic dimensions)
                    actual_shape = tuple(out.shape)
                    if actual_shape != output_shape:
                        # Check if it's just a dynamic dimension difference
                        can_match = all(
                            a == b or b == -1
                            for a, b in zip(actual_shape, output_shape)
                        )
                        if not can_match:
                            # Shape mismatch, allocate new tensor
                            out = None
                
                # Use the tensor or allocate new one
                if isinstance(out, torch.Tensor):
                    outputs[j] = out
                else:
                    # Allocate new output tensor
                    torch_dtype = datatype_to_torch_dtype(info.dtype)
                    outputs[j] = torch.empty(
                        output_shape,
                        dtype=torch_dtype,
                        device='cuda',
                        memory_format=torch.contiguous_format
                    )
            else:
                # Allocate new output tensor
                torch_dtype = datatype_to_torch_dtype(info.dtype)
                outputs.append(
                    torch.empty(
                        output_shape,
                        dtype=torch_dtype,
                        device='cuda',
                        memory_format=torch.contiguous_format
                    )
                )
        
        return outputs
    
    def _set_output_tensors(self, outputs: List[torch.Tensor]) -> None:
        """Set output tensor addresses in the context."""
        output_infos = self._io_info[1]
        
        for out, info in zip(outputs, output_infos):
            name = info.name
            
            # Ensure out is a torch.Tensor (handle omniback StdAny)
            if not isinstance(out, torch.Tensor):
                # Try to convert from DLPack or tvm_ffi
                if hasattr(out, '__dlpack__'):
                    try:
                        out = torch.from_dlpack(out)
                    except Exception:
                        pass
                elif _tvm_ffi_available and hasattr(out, 'to_dlpack'):
                    try:
                        out = torch.from_dlpack(out.to_dlpack())
                    except Exception:
                        pass
                
                # If still not a tensor, we can't use it
                if not isinstance(out, torch.Tensor):
                    raise TensorRTError(f"Output '{name}' is not a torch.Tensor: {type(out)}")
            
            # Ensure tensor is contiguous
            if not out.is_contiguous():
                out = out.contiguous()
            
            # Set tensor address
            success = self._context.set_tensor_address(name, out.data_ptr())
            if not success:
                raise TensorRTError(f"Failed to set output tensor address for '{name}'")
    
    def _execute(self) -> None:
        """Execute the inference."""
        # Get current CUDA stream
        stream = CUDAStreamManager.get_current_stream()
        
        # Execute async
        if hasattr(self._context, 'execute_async_v3'):
            success = self._context.execute_async_v3(stream.cuda_stream)
        elif hasattr(self._context, 'enqueue_v3'):
            success = self._context.enqueue_v3(stream.cuda_stream)
        else:
            raise TensorRTError("No supported execution method available")
        
        if not success:
            raise TensorRTError("TensorRT inference execution failed")
        
        # Synchronize on the stream to ensure completion
        stream.synchronize()
    
    def _set_result(self, io_dict: Any, outputs: List[torch.Tensor]) -> None:
        """Set the result in the IO dictionary."""
        # Remove input data to free memory
        if TASK_DATA_KEY in io_dict:
            del io_dict[TASK_DATA_KEY]
        
        # Set result
        if len(outputs) == 1:
            io_dict[TASK_RESULT_KEY] = outputs[0]
        else:
            io_dict[TASK_RESULT_KEY] = outputs
    
    def max(self) -> int:
        """Return the maximum batch size."""
        if self._io_info is None or len(self._io_info[0]) == 0:
            return 1
        max_dims = self._io_info[0][0].max
        return int(max_dims.d[0]) if max_dims.nbDims > 0 else 1
    
    def min(self) -> int:
        """Return the minimum batch size."""
        if self._io_info is None or len(self._io_info[0]) == 0:
            return 1
        min_dims = self._io_info[0][0].min
        return int(min_dims.d[0]) if min_dims.nbDims > 0 else 1
    
    def release(self) -> None:
        """
        Release all resources held by this backend.
        
        This method releases the TensorRT context and device memory.
        After calling this method, the backend cannot be used for inference.
        """
        # Release context
        self._context = None
        
        # Release device memory
        self._device_memory = None
        self._mem_size = 0
        
        # Release event
        self._input_finish_event = None
        
        # Note: We don't release the engine here because it may be shared
        # with other backends. The engine should be released separately.
        
        self._initialized = False
        
        logger.debug("PyTensorrtInferTensor resources released")
    
    def __del__(self):
        """Destructor to ensure resources are released."""
        try:
            self.release()
        except Exception:
            pass


# Register the backend
register_backend(
    "PyTensorrtInferTensor",
    PyTensorrtInferTensor,
    BackendMeta(
        name="PyTensorrtInferTensor",
        version="1.0.0",
        description="Python implementation of TensorRT inference backend. "
                    "Compatible with TensorRT 9.3 and TensorRT >= 10. "
                    "Minimum version: TensorRT 9.3.",
        tags=["tensorrt", "inference", "gpu", "python"],
        requires_cuda=True,
        supports_dynamic_shape=True,
        supports_multi_instance=True,
    )
)

__all__ = [
    "PyTensorrtEngine",
    "PyTensorrtInferTensor",
    "PyTensorrtConfig",
    "ProfileConfig",
]
