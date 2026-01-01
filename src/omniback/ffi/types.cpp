#include <unordered_map>
#include <mutex>

// #include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/function.h>
// #include "omniback/ffi/dict.h"
#include <tvm/ffi/reflection/registry.h>
#include "omniback/core/any.hpp"
#include <tvm/ffi/function.h>
#include <memory>
#include <string>
#include <unordered_map>
#include <tvm/ffi/container/array.h>
#include "omniback/ffi/event.h"
#include "omniback/ffi/dict.h"
#include "omniback/ffi/types.hpp"

// #include <tvm/ffi/container/tensor.h>

namespace omniback::ffi {
struct xx{};
tvm::ffi::ObjectRef example(tvm::ffi::ObjectRef obj) {
  return obj;
             // return 1;
            //  std::shared_ptr<std::unordered_map<std::string, omniback::any>>
            //      re = std::make_shared<
            //          std::unordered_map<std::string, omniback::any>>();
  // tvm::ffi::Array<
  //     std::shared_ptr<std::unordered_map<std::string, omniback::any>>>
  //     res;
  // return res;
  //  res->emplace(re);
  // return omniback::any(std::numeric_limits<uint32_t>::max());
}

int64_t& dlpack_exchange_api(){
  static int64_t api = 0;
  return api;
}

void set_dlpack_exchange_api(int64_t api){
  int64_t& sapi = dlpack_exchange_api();
  sapi = api;
}

// DLPackManagedTensorAllocator& env_allocator() {
//   static auto alloc = TVMFFIEnvGetDLPackManagedTensorAllocator();
//   return alloc;
// }

// void capture_env(tvm::ffi::TensorView x) {
//   auto alloc = env_allocator();

// }

// Static initialization block for FFI registration
// TVM_FFI_STATIC_INIT_BLOCK() {
//   namespace refl = tvm::ffi::reflection;

//   // Register object definition with constructor
//   refl::ObjectDef<AnyObj>()
//       .def(refl::init<>())
//       .def("type_name", [](const AnyObj* obj) {
//         return obj->data.type().name();
//       });
// }

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("omniback.example", example);
  refl::GlobalDef().def(
      "omniback.set_dlpack_exchange_api", set_dlpack_exchange_api);
}
// TVM_FFI_DLL_EXPORT_TYPED_FUNC(example, example)

} // namespace omniback::ffi
