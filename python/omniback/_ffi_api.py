
import tvm_ffi

from . import libinfo

ffi = tvm_ffi.load_module(libinfo.find_libomniback())

# this is a short cut to register all the global functions
tvm_ffi.init_ffi_api("omniback", __name__)


@tvm_ffi.register_object("omniback.FFIQueue")
class FFIQueue(tvm_ffi.Object):
    def __init__(self) -> None:
        """Construct the object."""
        # __ffi_init__ call into the refl::init<> registered
        # in the static initialization block of the extension library
        self.__ffi_init__()


ffi.Queue = FFIQueue


def default_queue(tag=""):
    return ffi.default_queue_one_arg(tag)


ffi.default_queue = default_queue
