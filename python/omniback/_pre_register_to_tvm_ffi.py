import tvm_ffi
import importlib

@tvm_ffi.register_global_func("om.get_function_from_py")
def get_py_func(module_name: str, function_name: str):
    module = importlib.import_module(module_name)
    func = getattr(module, function_name)
    return func


@tvm_ffi.register_global_func("om.get_attr")
def get_attr(obj, function_name: str):
    func = getattr(obj, function_name)
    return func


@tvm_ffi.register_global_func("om.get_attr_and_call")
def get_attr_and_call(obj, function_name: str, *args, **kwargs):
    method = getattr(obj, function_name)
    return method(*args, **kwargs)
