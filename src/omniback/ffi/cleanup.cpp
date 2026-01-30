#include "omniback/ffi/cleanup.h"
#include <tvm/ffi/reflection/registry.h>
#include <functional>
#include <queue>

#include "omniback/py/function_holder.hpp"

namespace om::ffi {
namespace {
auto& get_cleanup_registry() {
  static std::queue<std::function<void()>> registry;
  return registry;
}

void internal_cleanup(){
    auto &reg = get_cleanup_registry();
    while(!reg.empty()){
        auto func = std::move(reg.front());
        reg.pop();
        func();
    }
}
}



void  atexit_register(std::function<void()> clear_func){
        auto &reg = get_cleanup_registry();
        reg.push(std::move(clear_func));   
    }

TVM_FFI_STATIC_INIT_BLOCK() {
  om::py::pycall("atexit", "register", tvm::ffi::TypedFunction<void()>(internal_cleanup));
}
}