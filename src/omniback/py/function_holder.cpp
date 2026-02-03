#include "omniback/py/function_holder.hpp"
#include <tvm/ffi/error.h>

namespace om::py {

tvm::ffi::Function  get_py_func(tvm::ffi::String module_name, tvm::ffi::String func_name){
    auto f = tvm::ffi::Function::GetGlobalRequired("om.get_py_func");
    return f(module_name, func_name).cast<tvm::ffi::Function>();
}

}