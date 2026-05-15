#include <unordered_map>
#include <mutex>

#include <memory>
#include <vector>
#include <string>
#include <utility>
#include <algorithm>

#include <tvm/ffi/error.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include "omniback/ffi/any_wrapper.h"
#include "tvm/ffi/container/variant.h"
#include "tvm/ffi/extra/stl.h"
#include <tvm/ffi/type_traits.h>

#include "omniback/ffi/event.h"
#include "omniback/helper/timer.hpp"

namespace om::ffi {

EventObj::EventObj(uint32_t num)
    : num_task(num), starttime_(om::helper::now()) {}

float EventObj::time_passed() {
  return om::helper::time_passed(starttime_);
}

namespace tf = tvm::ffi;
namespace refl = tvm::ffi::reflection;

TVM_FFI_STATIC_INIT_BLOCK() {
  refl::ObjectDef<EventObj>()
      .def(refl::init<uint32_t>())
      .def("wait", [](EventObj* self, uint32_t timeout_ms) {
        return self->wait(timeout_ms);
      });
}

}  // namespace om::ffi
