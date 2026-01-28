#include <tvm/ffi/function.h>
#include <tvm/ffi/string.h>





namespace om {
tvm::ffi::Function  get_function_from_py(tvm::ffi::String module_name, tvm::ffi::String func_name);
}