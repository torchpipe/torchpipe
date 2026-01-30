#include <tvm/ffi/function.h>
#include <tvm/ffi/string.h>
// #include <tvm/ffi/container/tuple.h>





namespace om::py {
tvm::ffi::Function  get_py_func(tvm::ffi::String module_name, tvm::ffi::String func_name);

template <typename... Args>
tvm::ffi::Any pycall(tvm::ffi::String module_name,
                     tvm::ffi::String func_name,
                     Args&&... args) {
    auto py_func = get_py_func(module_name, func_name);
    return py_func(std::forward<Args>(args)...); 
}


}