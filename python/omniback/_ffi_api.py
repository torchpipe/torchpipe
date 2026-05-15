"""FFI API for Omniback - provides Python bindings to C++ core."""

from __future__ import annotations

import ctypes
import logging
from typing import Any, overload

import tvm_ffi

from . import libinfo
from . import _pre_register_to_tvm_ffi  # noqa: F401

logger = logging.getLogger(__name__)

# Load the core library
libomniback = ctypes.CDLL(
    str(libinfo.find_libomniback()),
    ctypes.RTLD_GLOBAL
)
logger.info("Loaded omniback library: %s", libomniback)

# Initialize FFI API
tvm_ffi.init_ffi_api("omniback", __name__)


@tvm_ffi.register_object("omniback.Queue")
class FFIQueue(tvm_ffi.Object):
    """FFI wrapper for omniback Queue."""

    def __init__(self) -> None:
        """Construct the Queue object."""
        self.__ffi_init__()


@tvm_ffi.register_object("omniback.Dict")
class OmDict(tvm_ffi.Object):
    """FFI wrapper for omniback Dict - a string-keyed dictionary."""

    def __init__(self, data: dict[str, Any] | None = None) -> None:
        """Construct an omniback.Dict.

        Args:
            data: Optional initial dictionary data
        """
        if data is None:
            self.__ffi_init__({})
        else:
            if not isinstance(data, dict):
                raise TypeError("data must be a dict")
            self.__ffi_init__(data)


class _PyDictWrapper:
    """Internal helper to bridge Python dict and OmDict."""

    def __init__(self, dict_obj: dict, om_dict: OmDict):
        self.dict_obj = dict_obj
        self.om_dict = om_dict

    def callback(self) -> None:
        """Sync data from OmDict back to Python dict."""
        self.dict_obj.pop("result", None)
        for k, v in self.om_dict.items():
            self.dict_obj[k] = v


@tvm_ffi.register_object("omniback.Backend")
class Backend(tvm_ffi.Object):
    """FFI wrapper for omniback Backend."""

    def __init__(self, params: dict[str, str] | None = None, options: Any = None) -> None:
        """Construct and initialize a Backend.

        Args:
            params: Backend parameters
            options: Backend options
        """
        self.__ffi_init__()
        self._init(params or {}, options)

    def init(self, params: dict[str, str] | None = None, options: Any = None) -> Any:
        """Initialize/reinitialize the backend.

        Args:
            params: Backend parameters
            options: Backend options

        Returns:
            Initialization result
        """
        return self._init(params or {}, options)

    @overload
    def __call__(self, ios: list[OmDict]) -> None: ...
    @overload
    def __call__(self, ios: dict[str, Any]) -> None: ...
    @overload
    def __call__(self, ios: OmDict) -> None: ...

    def __call__(self, ios: list[OmDict] | dict[str, Any] | OmDict) -> None:
        """Execute the backend with flexible input types.

        This method provides a convenient interface for executing the backend
        with various input formats. The input is automatically converted to
        a list of OmDict before forwarding to the backend.

        Args:
            ios: Input data in one of the following formats:
                - list[OmDict]: A batch of OmDict objects (direct passthrough)
                - dict: A Python dictionary (converted to OmDict internally)
                - OmDict: A single OmDict object (wrapped in list)

        Raises:
            TypeError: If input type is not supported
            RuntimeError: If dict input contains 'event' key (use OmDict for async)

        Examples:
            >>> backend = omniback.init("Identity", {}, None, None)
            >>> backend([omniback.Dict({"data": "test"})])  # Batch input
            >>> backend({"data": "test"})  # Dict input
            >>> backend(omniback.Dict({"data": "test"}))  # OmDict input
        """
        if isinstance(ios, list):
            if not ios:
                raise ValueError("Input list cannot be empty")
            if not all(isinstance(io, OmDict) for io in ios):
                raise TypeError("All items in list must be omniback.Dict")
            self.forward(ios)
        elif isinstance(ios, dict):
            if "event" in ios:
                raise RuntimeError("Use omniback.Dict for async execution with event")
            input_dict = OmDict(ios)
            input_dict.callback = _PyDictWrapper(ios, input_dict).callback
            self.forward([input_dict])
        elif isinstance(ios, OmDict):
            self.forward([ios])
        else:
            raise TypeError(
                f"Input must be List[omniback.Dict], omniback.Dict, or dict, "
                f"got {type(ios).__name__}"
            )


@tvm_ffi.register_object("omniback.Event")
class Event(tvm_ffi.Object):
    """FFI wrapper for omniback Event - used for async execution."""

    def __init__(self, num: int = 1) -> None:
        """Construct an Event.

        Args:
            num: Number of events
        """
        self.__ffi_init__(num)


Queue = FFIQueue
