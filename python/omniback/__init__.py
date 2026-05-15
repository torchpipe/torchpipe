

import atexit
import os

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
except ImportError:
    pass

try:
    import torch
    from packaging import version

    if version.parse(torch.__version__) < version.parse("2.4.0"):
        # skip compilation step of tvm_ffi: https://github.com/apache/tvm-ffi/issues/381
        os.environ["TVM_FFI_DISABLE_TORCH_C_DLPACK"] = "1"
except ImportError:
    pass

# isort: off
import tvm_ffi
from . import utils
from .parser import parse, parse_group, init_from_file, pipe
from . import libinfo
from . import _ffi_api as ffi

from ._ffi_api import libomniback


from ._ffi_api import Queue, Event, Backend
from ._ffi_api import OmDict as Dict


def compiled_with_cxx11_abi():
    return ffi.use_cxx11_abi()

def create(name, register_name=None):
    return ffi.create(name, register_name)


def default_queue(tag=""):
    return ffi.default_queue_one_arg(tag)


def default_page_table(tag=""):
    return ffi.default_page_table(tag)

def init(name, params={}, options=None, register_name=None):
    return ffi.init(name, params, options, register_name)


import logging
logger = logging.getLogger(__name__)

def register(name, object_or_type):
    import inspect

    if not isinstance(name, str):
        raise TypeError(f"Registration name must be a string, got {type(name).__name__}")
    
    if not name:
        raise ValueError("Registration name cannot be empty")

    try:
        if isinstance(object_or_type, type):
            ins_type = object_or_type

            if not hasattr(ins_type, "__init__"):
                raise TypeError(f"Class '{ins_type.__name__}' has no __init__ method")

            init_signature = inspect.signature(ins_type.__init__)
            params = list(init_signature.parameters.values())

            for param in params[1:]:
                if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                    continue
                if param.default is inspect.Parameter.empty:
                    error_msg = (
                        f"Class '{ins_type.__name__}' cannot be default-constructed: "
                        f"parameter '{param.name}' has no default value. "
                        f"Please ensure all __init__ parameters have default values "
                        f"or provide a factory function directly."
                    )
                    logger.error(error_msg)
                    raise TypeError(error_msg)

            def create_instance():
                return ins_type()
        else:
            ins_type = type(object_or_type)
            create_instance = object_or_type

        init_func = getattr(ins_type, "init", None)
        forward_func = getattr(ins_type, "forward", None)
        max_func = getattr(ins_type, "max", None)
        min_func = getattr(ins_type, "min", None)
        
        if forward_func is None:
            logger.warning(
                "Registering backend '%s' without a 'forward' method. "
                "This backend may not function correctly.",
                name
            )
        
        logger.debug("Registering backend '%s' with type '%s'", name, ins_type.__name__)
        
        return ffi.register(name, create_instance, init_func, forward_func, max_func, min_func)
    except Exception as e:
        logger.error("Failed to register backend '%s': %s", name, e, exc_info=True)
        raise


assert atexit.register(ffi.cleanup)


def get(name):
    return ffi.get(name)


try:
    # type: ignore[import-not-found]
    from ._version import __version__, __version_tuple__
except ImportError:
    __version__ = "0.0.0.dev0"
    __version_tuple__ = (0, 0, 0, "dev0", "unknown")


# isort: on


def get_library_dir():
    return os.path.dirname(libinfo.find_libomniback())


def get_include_dirs(with_tvm_ffi=False):
    return libinfo.include_paths(with_tvm_ffi=with_tvm_ffi)


def extra_include_paths():
    return libinfo.include_paths(),


def extra_ldflags():
    return [f"-L{get_library_dir()}", '-lomniback'],


__all__ = [
    "Any", "Dict", "Backend", "Event", "Queue",
    "create", "register", "parse", "parse_group", "init", "get",
    "pipe", "default_queue", "default_page_table",
    "compiled_with_cxx11_abi",
    "extra_include_paths", "extra_ldflags",
    "utils", "libinfo", "ffi"
]
