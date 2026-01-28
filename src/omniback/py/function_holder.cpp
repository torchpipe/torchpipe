#include "omniback/py/function_holder.hpp"
#include <tvm/ffi/error.h>

namespace om {

tvm::ffi::Function  get_function_from_py(tvm::ffi::String module_name, tvm::ffi::String func_name){
    static auto f = tvm::ffi::Function::GetGlobalRequired("om.get_function_from_py");
    return f(module_name, func_name).cast<tvm::ffi::Function>();
}

}