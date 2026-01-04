
import struct
import tvm_ffi

from . import libinfo

# try:
#     import torch
# except:
#     pass

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


# @tvm_ffi.register_object("omniback.StdAny")
# class StdAny(tvm_ffi.Object):
#     def __init__(self) -> None:
#         """Construct a omniback.StdAny."""
#         self.__ffi_init__()  # todo; repair
    
#     def as_torch(self):
#         return omniback.ffi.from_dlpack(self.to_tensor())
    
#     def __dlpack__(self, stream):
#         self.data = self.to_tensor()
#         return self.data.__dlpack__(stream)

#     def __dlpack_device__(self):
#         return self.data.__dlpack_device__()


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
            if "event" in ios:
                raise RuntimeError("Use omniback.Dict for async")
            input = OmDict(ios)
            input.callback = _PyDictWrapper(ios, input).callback
            self.forward([input])
            del input
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
