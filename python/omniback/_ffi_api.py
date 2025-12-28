
import struct
import tvm_ffi

from . import libinfo

_C = tvm_ffi.load_module(libinfo.find_libomniback())

# this is a short cut to register all the global functions
tvm_ffi.init_ffi_api("omniback", __name__)


@tvm_ffi.register_object("omniback.Queue")
class FFIQueue(tvm_ffi.Object):
    def __init__(self) -> None:
        """Construct the object."""
        # __ffi_init__ call into the refl::init<> registered
        # in the static initialization block of the extension library
        self.__ffi_init__()


@tvm_ffi.register_object("omniback.Dict")
class OmDict(tvm_ffi.Object):
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


class _PyDictWrapper:
    def __init__(self, dict_obj: dict, om_dict: OmDict):
        self.dict_obj = dict_obj
        self.om_dict = om_dict

    def callback(self):
        self.dict_obj.pop("result", None)
        for k, v in self.om_dict.items():
            self.dict_obj[k] = v


@tvm_ffi.register_object("omniback.Backend")
class Backend(tvm_ffi.Object):
    def __init__(self, params={}, options=None) -> None:
        """Construct a omniback.Backend."""
        self.__ffi_init__()
        self._init(params, options)
    
    def init(self, params={}, options=None):
        """Initialize the backend with the given parameters and options."""
        return self._init(params, options)
    
    def __call__(self, ios):
        """Execute the backend with the given input dictionary."""
        if isinstance(ios, list):
            assert all(isinstance(io, OmDict)
                       for io in ios), "Please use List[omniback.Dict] as input"
            self.forward(ios)
        elif isinstance(ios, dict):
            input = OmDict(ios)
            input.callback = _PyDictWrapper(ios, input).callback
            self.forward([input])
        elif isinstance(ios, OmDict):
            self.forward([ios])
        else:
            raise TypeError("Input must be List[omniback.Dict] or omniback.Dict or dict")

        # self._call(input_dict)


@tvm_ffi.register_object("omniback.Event")
class Event(tvm_ffi.Object):
    def __init__(self, num=1) -> None:
        """Construct a omniback.Event."""
        self.__ffi_init__(num)


Queue = FFIQueue
# _C.Queue = FFIQueue


def default_queue(tag=""):
    return _C.default_queue_one_arg(tag)


def default_page_table(tag=""):
    return _C.default_page_table(tag)

# _C.default_queue = default_queue
