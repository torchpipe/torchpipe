

from . import _ffi_api
import sys
import os
from pathlib import Path
import tvm_ffi

# isort: off
from . import libinfo
# libinfo.load_lib_ctypes('omniback', 'omniback', "RTLD_GLOBAL")
ffi = tvm_ffi.load_module(libinfo.find_libomniback())
from . import omniback_py as _C # noqa F401
sys.modules['omniback._C'] = _C


@tvm_ffi.register_object("omniback.FFIQueue")
class FFIQueue(tvm_ffi.Object):
    def __init__(self) -> None:
        """Construct the object."""
        # __ffi_init__ call into the refl::init<> registered
        # in the static initialization block of the extension library
        self.__ffi_init__()


ffi.Queue = FFIQueue
# isort: on

try:
    # type: ignore[import-not-found]
    from ._version import __version__, __version_tuple__
except ImportError:
    __version__ = "0.0.0.dev0"
    __version_tuple__ = (0, 0, 0, "dev0", "unknown")


# isort: off
from ._C import Any, Dict, Backend, Event, create, register, get, TypedDict, print, timestamp, init
from ._C import (Queue, default_queue, default_page_table)
from ._C import TASK_DATA_KEY, TASK_RESULT_KEY, TASK_MSG_KEY, TASK_REQUEST_ID_KEY,TASK_EVENT_KEY,TASK_WAITING_EVENT_KEY


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
    return [f"-L{get_library_dir()}", '-lomniback',
                 f'{_C.__file__}'],

__all__ = ["Any", "Dict", 'Backend', 'Event', 'create', 'create', 'register', 'parse',
           'init', 'get', 'default_page_table', "timestamp", "pipe", 'init', 'load_kwargs', "_C"]

__all__.extend(['Queue', 'default_queue'])

__all__.extend(['print', 'utils', 'libinfo', 'ffi'])

__all__.extend(['print', 'utils', 'libinfo', 'ffi',
               'extra_include_paths', 'extra_ldflags'])
