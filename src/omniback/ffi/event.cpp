#include <unordered_map>
#include <mutex>

#include <memory>
#include <vector>
#include <string>
#include <utility> // for std::pair
#include <algorithm> // for std::find

#include <tvm/ffi/error.h>

#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ffi/function.h>
#include "omniback/ffi/any_wrapper.h"
// #include "tvm/ffi/container/map.h"
#include "tvm/ffi/container/variant.h"
#include "tvm/ffi/extra/stl.h"
#include <tvm/ffi/type_traits.h>
#include <tvm/ffi/reflection/registry.h>

#include "omniback/ffi/event.h"
#include "omniback/helper/timer.hpp"

namespace omniback::ffi {

EventObj::EventObj(uint32_t num)
    : num_task(num), starttime_(omniback::helper::now()) {}
float EventObj::time_passed() {
  return omniback::helper::time_passed(starttime_);
}

namespace tf = tvm::ffi;
namespace refl = tvm::ffi::reflection;


TVM_FFI_STATIC_INIT_BLOCK() {
  refl::ObjectDef<EventObj>()
      .def(refl::init<uint32_t>())
      // .def("wait", [](EventObj* self) { return self->wait(); })
      .def("wait", [](EventObj* self, uint32_t timeout_ms) {
        return self->wait(timeout_ms);
      });
}

} // namespace omniback::ffi

namespace tvm::ffi {
// template <>
// inline constexpr bool use_default_type_traits_v<
//     std::shared_ptr<std::unordered_map<std::string, omniback::any>>> = false;

// template <>
// struct TypeTraits<
//     std::shared_ptr<std::unordered_map<std::string, omniback::any>>>
//     : public TypeTraitsBase {
//  public:
//   static constexpr bool storage_enabled = false;
//   using Self = std::shared_ptr<std::unordered_map<std::string, omniback::any>>;
//   using DictObj = omniback::ffi::DictObj;

//   TVM_FFI_INLINE static void
//           MoveToAny(Self&& src, TVMFFIAny* result) {
//     auto data = tvm::ffi::make_object<DictObj>(std::move(src));
//     tvm::ffi::TypeTraits<DictObj*>::MoveToAny(data.get(), result);
//   }

//   TVM_FFI_INLINE static std::string TypeStr() {
//     return "omniback::Dict";
//   }
//   TVM_FFI_INLINE static std::string TypeSchema() {
//     return R"({"type":"omniback::Dict"})";
//   }
// };

}; // namespace tvm::ffi

// namespace tvm::ffi {
// using Dict = omniback::ffi::Dict; 
// template <>
// inline constexpr bool use_default_type_traits_v<Dict> = false;

// // Allow auto conversion from Map to Dict, but not from Dict to
// // Map
// template <>
// struct TypeTraits<Dict>
//     : public tvm::ffi::ObjectRefWithFallbackTraitsBase<
//           Dict,
//           tvm::ffi::Map<tvm::ffi::String, tvm::ffi::Any>> {
//   TVM_FFI_INLINE static Dict ConvertFallbackValue(
//       tvm::ffi::Map<tvm::ffi::String, tvm::ffi::Any> src) {
//     return Dict(std::move(src));
//   }
// };
// }

