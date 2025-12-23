

import sys
import os
from pathlib import Path
import tvm_ffi

# isort: off
from . import libinfo

# from . import _ffi_api

from . import _ffi_api as ffi
from ._ffi_api import _C


from ._ffi_api import Dict, Queue, default_queue, Event, Backend

def create(name, register_name=None):
    return ffi.create(name, register_name)


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
