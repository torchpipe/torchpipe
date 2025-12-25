

import sys
import os
from pathlib import Path
import tvm_ffi

# isort: off
from . import libinfo

# from . import _ffi_api

from . import _ffi_api as ffi
from ._ffi_api import _C


from ._ffi_api import  Queue, default_queue, Event, Backend
from ._ffi_api import OmDict as Dict

def create(name, register_name=None):
    return ffi.create(name, register_name)

def init(name, params={}, options=None, register_name=None):
    return ffi.init(name, params, options, register_name)


def register(name, object_or_type):
    if isinstance(object_or_type, type):
        raise NotImplementedError(
            "Registering from type is not implemented yet")
    else:
        # init_func = getattr(object_or_type, "init", None)
        # forward_func = getattr(object_or_type, "forward", None)
        # max_func = getattr(object_or_type, "max", None)
        # min_func = getattr(object_or_type, "min", None)
        ins_type = type(object_or_type)
        init_func = getattr(ins_type, "init", None)
        forward_func = getattr(ins_type, "forward", None)
        max_func = getattr(ins_type, "max", None)
        min_func = getattr(ins_type, "min", None)
    return ffi.register(name, object_or_type, init_func, forward_func, max_func, min_func)

import atexit
assert atexit.register(ffi.cleanup)

def get(name):
    return ffi.get(name)

try:
    # type: ignore[import-not-found]
    from ._version import __version__, __version_tuple__
except ImportError:
    __version__ = "0.0.0.dev0"
    __version_tuple__ = (0, 0, 0, "dev0", "unknown")


from .parser import parse, init_from_file, pipe

from . import utils
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
           'init', 'get', 'default_page_table', "timestamp", "pipe", 'init', 'load_kwargs', "_C"]

__all__.extend(['Queue', 'default_queue'])

__all__.extend(['print', 'utils', 'libinfo', 'ffi'])

__all__.extend(['print', 'utils', 'libinfo', 'ffi',
               'extra_include_paths', 'extra_ldflags'])
