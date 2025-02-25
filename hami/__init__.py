from .version import __version__
from .commands import get_cmake_dir, get_includes, get_pkgconfig_dir, get_library_dir, get_root, get_C_path


from hami._C import any, dict, Backend, Event, create, register, init
# from hami import pybackend
from .parser import parse_from_file

__all__ = ["any", "dict", 'Backend', 'Event', 'create', 'create', 'register', 'parse_from_file', 'init']