
from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("hami-core")
except PackageNotFoundError:
    # package is not installed
    pass



from .commands import get_cmake_dir, get_includes, get_pkgconfig_dir, get_library_dir, get_root, get_C_path


from hami._C import Any, Dict, Backend, Event, create, register, init, get, TypedDict, print

from hami._C import (Queue, default_queue, default_page_table)

from hami._C import TASK_DATA_KEY, TASK_RESULT_KEY, TASK_MSG_KEY, TASK_REQUEST_ID_KEY,TASK_EVENT_KEY,TASK_WAITING_EVENT_KEY
# from hami import pybackend
from .parser import parse, init_from_file

__all__ = ["Any", "Dict", 'Backend', 'Event', 'create', 'create', 'register', 'parse', 'init', 'get', 'default_page_table']

__all__.extend(['Queue', 'default_queue'])

__all__.extend(['print'])

