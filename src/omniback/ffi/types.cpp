#include <unordered_map>
#include <mutex>

#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include "omniback/core/any.hpp"
#include <memory>
#include <string>
#include <tvm/ffi/container/array.h>
#include "omniback/ffi/event.h"
#include "omniback/ffi/dict.h"
#include "omniback/ffi/types.hpp"

namespace om::ffi {

tvm::ffi::ObjectRef example(tvm::ffi::ObjectRef obj) {
  return obj;
}

int64_t& dlpack_exchange_api(){
  static int64_t api = 0;
  return api;
}

void set_dlpack_exchange_api(int64_t api){
  int64_t& sapi = dlpack_exchange_api();
  sapi = api;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("omniback.example", example);
  refl::GlobalDef().def(
      "omniback.set_dlpack_exchange_api", set_dlpack_exchange_api);
}

} // namespace om::ffi
