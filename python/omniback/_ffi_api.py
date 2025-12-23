
import struct
import tvm_ffi

from . import libinfo

_C = tvm_ffi.load_module(libinfo.find_libomniback())

# this is a short cut to register all the global functions
tvm_ffi.init_ffi_api("omniback", __name__)


@tvm_ffi.register_object("omniback.FFIQueue")
class FFIQueue(tvm_ffi.Object):
    def __init__(self) -> None:
        """Construct the object."""
        # __ffi_init__ call into the refl::init<> registered
        # in the static initialization block of the extension library
        self.__ffi_init__()


@tvm_ffi.register_object("omniback.Dict")
class Dict(tvm_ffi.Object):
    def __init__(self, data=None) -> None:
        """Construct a omniback.Dict."""
        if data is None:
            self.__ffi_init__({})  # todo: c++ side should handle default
        else:
            assert isinstance(data, dict)
            self.__ffi_init__(data)

    def __repr__(self) -> str:
        if self.__chandle__() == 0:
            return f"{type(self).__name__}(chandle=None)"
        items_repr = ", ".join(f"{k!r}: {v!r}" for k, v in self.items())
        return f"{{{items_repr}}}"




@tvm_ffi.register_object("omniback.Backend")
class Backend(tvm_ffi.Object):
    def __init__(self, data=None) -> None:
        """Construct a omniback.Backend."""
        self.__ffi_init__()




@tvm_ffi.register_object("omniback.Event")
class Event(tvm_ffi.Object):
    def __init__(self, num=1) -> None:
        """Construct a omniback.Event."""
        self.__ffi_init__(num) 
         


Queue = FFIQueue
# _C.Queue = FFIQueue


def default_queue(tag=""):
    return _C.default_queue_one_arg(tag)


# _C.default_queue = default_queue
