import logging
import tvm_ffi
import importlib

logger = logging.getLogger(__name__)


@tvm_ffi.register_global_func("om.get_py_func")
def get_py_func(module_name: str, function_name: str):
    try:
        module = importlib.import_module(module_name)
        func = getattr(module, function_name)
        return func
    except ImportError as e:
        logger.error("Failed to import module '%s': %s", module_name, e)
        raise RuntimeError(f"Module '{module_name}' not found: {e}") from e
    except AttributeError as e:
        logger.error("Attribute '%s' not found in module '%s': %s", function_name, module_name, e)
        raise RuntimeError(f"Function '{function_name}' not found in module '{module_name}'") from e
    except Exception as e:
        logger.error("Unexpected error getting function '%s' from module '%s': %s", function_name, module_name, e)
        raise


@tvm_ffi.register_global_func("om.get_attr")
def get_attr(obj, function_name: str):
    try:
        func = getattr(obj, function_name)
        return func
    except AttributeError as e:
        obj_type = type(obj).__name__
        logger.error("Attribute '%s' not found on object of type '%s': %s", function_name, obj_type, e)
        raise RuntimeError(f"Attribute '{function_name}' not found on object of type '{obj_type}'") from e
    except Exception as e:
        obj_type = type(obj).__name__
        logger.error("Unexpected error getting attribute '%s' from '%s': %s", function_name, obj_type, e)
        raise


@tvm_ffi.register_global_func("om.get_attr_and_call")
def get_attr_and_call(obj, function_name: str, *args, **kwargs):
    try:
        method = getattr(obj, function_name)
        return method(*args, **kwargs)
    except AttributeError as e:
        obj_type = type(obj).__name__
        logger.error("Attribute '%s' not found on object of type '%s': %s", function_name, obj_type, e)
        raise RuntimeError(f"Method '{function_name}' not found on object of type '{obj_type}'") from e
    except Exception as e:
        obj_type = type(obj).__name__
        logger.error("Error calling method '%s' on '%s': %s", function_name, obj_type, e)
        raise
