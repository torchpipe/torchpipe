from typing import Any, Dict, List, Optional, Union

import ctypes, os

import omniback
ctypes.CDLL(omniback._C.__file__, mode=ctypes.RTLD_GLOBAL)


import torch

from importlib.metadata import version

__version__ = version("torchpipe")

ctypes.CDLL(os.path.join(os.path.dirname(__file__), "native.so"), mode=ctypes.RTLD_GLOBAL)

from . import native, image, trt

try:
    from . import mat
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

pipe = omniback.pipe
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
