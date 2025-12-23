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

namespace omniback::ffi {
struct xx{};
omniback::any example() {
  // return 1;
  std::shared_ptr<std::unordered_map <std::string,
      omniback::any>> re = std::make_shared<std::unordered_map <std::string,
      omniback::any>>();
  // tvm::ffi::Array<
  //     std::shared_ptr<std::unordered_map<std::string, omniback::any>>>
  //     res;
  // return res;
  //  res->emplace(re);
  return omniback::any(std::numeric_limits<uint32_t>::max());
}

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
  
}
// TVM_FFI_DLL_EXPORT_TYPED_FUNC(example, example)

} // namespace omniback::ffi
