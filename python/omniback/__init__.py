

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
except:
    pass

try:
    import torch
    from packaging import version

    if version.parse(torch.__version__) < version.parse("2.4.0"):
        # skip compilation step of tvm_ffi: https://github.com/apache/tvm-ffi/issues/381
        os.environ["TVM_FFI_DISABLE_TORCH_C_DLPACK"] = "1"
except:
    pass

# isort: off
import tvm_ffi
from . import utils
from .parser import parse, init_from_file, pipe
from . import libinfo
from . import _ffi_api as ffi
from ._ffi_api import _C


from ._ffi_api import Queue, default_queue, Event, Backend
from ._ffi_api import OmDict as Dict


def create(name, register_name=None):
    return ffi.create(name, register_name)


def init(name, params={}, options=None, register_name=None):
    return ffi.init(name, params, options, register_name)


def register(name, object_or_type):
    import inspect

    if isinstance(object_or_type, type):
        ins_type = object_or_type

        init_signature = inspect.signature(ins_type.__init__)
        params = list(init_signature.parameters.values())

        for param in params[1:]:
            if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                continue
            if param.default is inspect.Parameter.empty:
                raise TypeError(
                    f"Class '{ins_type.__name__}' cannot be default-constructed: "
                    f"parameter '{param.name}' has no default value"
                )

        def creator_or_instance(): return ins_type()
    else:
        ins_type = type(object_or_type)
        creator_or_instance = object_or_type
    init_func = getattr(ins_type, "init", None)
    forward_func = getattr(ins_type, "forward", None)
    max_func = getattr(ins_type, "max", None)
    min_func = getattr(ins_type, "min", None)
    return ffi.register(name, creator_or_instance, init_func, forward_func, max_func, min_func)


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


def get_include_dirs():
    return libinfo.include_paths()


def extra_include_paths():
    return libinfo.include_paths(),


def extra_ldflags():
    return [f"-L{get_library_dir()}", '-lomniback'],


__all__ = ["Any", "Dict", 'Backend', 'Event', 'create', 'create', 'register', 'parse',
           'init', 'get',  "timestamp", "pipe", 'init', 'load_kwargs', "_C"]

__all__.extend(['Queue', 'default_queue'])

__all__.extend(['print', 'utils', 'libinfo', 'ffi'])

__all__.extend(['print', 'utils', 'libinfo', 'ffi',
               'extra_include_paths', 'extra_ldflags'])
