import ctypes, os

import omniback

import tvm_ffi
# ctypes.CDLL(omniback._C.__file__, mode=ctypes.RTLD_GLOBAL)

import torch

from importlib.metadata import version

__version__ = version("torchpipe")

ctypes.CDLL(os.path.join(os.path.dirname(__file__), "native.so"), mode=ctypes.RTLD_GLOBAL)

try:
    image = tvm_ffi.load_module(os.path.join(
        os.path.dirname(__file__), "image.so"))
except ImportError:
    print(f'nvjpeg related backends not loaded')
try:
    trt = tvm_ffi.load_module(os.path.join(
        os.path.dirname(__file__), "trt.so"))
except ImportError:
    print(f'trt related backends not loaded: ', os.path.dirname(__file__), "trt.so")
try:
    # from . import mat
    mat = tvm_ffi.load_module(os.path.join(
        os.path.dirname(__file__), "mat.so"))

except ImportError:
    print(f'opencv related backends not loaded')



from . import utils
if (omniback._C.use_cxx11_abi() != torch._C._GLIBCXX_USE_CXX11_ABI):
    info = f"Incompatible C++ ABI detected. Please re-install PyTorch/Torchpipe or omniback with the same C++ ABI. "
    info += "omniback CXX11_ABI = {}, torch CXX11_ABI = {}. ".format(
        omniback._C.use_cxx11_abi(), torch._C._GLIBCXX_USE_CXX11_ABI)
    info += f"""\nFor omniback, you can use 
        pip3 install omniback --platform manylinux2014_x86_64 --only-binary=:all:   --target `python3 -c "import site; print(site.getsitepackages()[0])"` 
        to install the pre-cxx11 abi version. Or use `USE_CXX11_ABI={int(not omniback._C.use_cxx11_abi())} pip install -e .` to rebuild omniback.
    """
    raise RuntimeError(info)



torch.cuda.init()

torch.set_num_threads(torch.get_num_threads())

import omniback as omni #
pipe = omniback.pipe
Dict = omniback.Dict
register = omniback.register
# class pipe:
#     """python interface for the c++ library. A simple wrapper for :ref:`Interpreter <Interpreter>` . Usage:

#     .. code-block:: python

#         models = pipe({"model":"model_bytes...."})
#         input = {'data':torch.from_numpy(...)}
#         result : torch.Tensor = input['result']

#     """

#     def __init__(self, config: Union[Dict[str, str], Dict[str, Dict[str, str]], str]):
#         """init with configuration.

#         :param config: toml file and plain dict are supported. These parameters will be passed to all the backends involved.
#         :type config: Dict[str, str] | Dict[str,Dict[str, str]] | str
#         """
#         self.Interpreter = omniback.create("Interpreter")
#         if not config:
#             raise RuntimeError(f"empty config : {config}")
#         if isinstance(config, dict):
#             for k, v in config.items():
#                 if isinstance(v, dict):
#                     for k2, v2 in v.items():
#                         if not isinstance(v, (bytes, str)):
#                             config[k][k2] = str(v2)  # .encode("utf8")
#                 else:
#                     if not isinstance(v, (bytes, str)):
#                         config[k] = str(v)  # .encode("utf8")
#             self._init(config)
#         else:
#             self._init_from_toml(config)

#     def _init_from_toml(self, toml_path):
#         self.config = omniback.parse(toml_path)
#         return self.Interpreter.init(self.config)

#     def _init(self, config):
#         self.config = config
#         return self.Interpreter.init(config)

#     def __call__(
#         self,
#         data: Optional["Dict[str, Any] | List[Dict[str, Any]]"] = None,
#         **kwargs: Any,
#     ) -> Union[None, Any]:
#         """thread-safe inference. The input could be a single dict, a list of dict for multiple inputs, or raw key-value pairs.

#         :param data: input dict(s), defaults to None
#         :type data: `Dict[str, Any] | List[Dict[str, Any]]`, optional
#         :return: None if data exists, else Any.
#         :rtype: Union[None, Any]
#         """
#         if isinstance(data, list):
#             if len(kwargs):
#                 for di in data:
#                     di.update(kwargs)
#             self.Interpreter(data)
#         elif isinstance(data, dict):
#             data.update(kwargs)
#             self.Interpreter(data)
#         else:
#             raise RuntimeError(f"invalid input: {type(data)}")
#             data = {}
#             data.update(kwargs)
#             self.Interpreter(data)
#             return data[TASK_RESULT_KEY]

#     def max(self):
#         return self.Interpreter.max()

#     def min(self):
#         return self.Interpreter.min()

#     def __del__(self):
#         self.Interpreter = None


if hasattr(torch.Tensor, "__dlpack_c_exchange_api__"):
    # type: ignore[attr-defined]
    api_attr = torch.Tensor.__dlpack_c_exchange_api__
    if api_attr:
        # PyCapsule - extract the pointer as integer
        pythonapi = ctypes.pythonapi
        # Set restype to c_size_t to get integer directly (avoids c_void_p quirks)
        pythonapi.PyCapsule_GetPointer.restype = ctypes.c_size_t
        pythonapi.PyCapsule_GetPointer.argtypes = [
            ctypes.py_object, ctypes.c_char_p]
        capsule_name = b"dlpack_exchange_api"
        api_ptr = pythonapi.PyCapsule_GetPointer(api_attr, capsule_name)
        assert api_ptr != 0, "API pointer from PyCapsule should not be NULL"
        omniback.ffi.set_dlpack_exchange_api(api_ptr)
