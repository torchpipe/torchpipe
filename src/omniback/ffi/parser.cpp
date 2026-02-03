// #include <unordered_map>
// #include <string>
// #include <functional>

#include "omniback/core/group.hpp"
#include "omniback/py/function_holder.hpp"
#include "omniback/ffi/cleanup.h"
#include "omniback/helper/base_logging.hpp"

#include <tvm/ffi/reflection/registry.h>
// #include <tvm/ffi/function.h>
// #include <tvm/ffi/string.h>
#include <tvm/ffi/extra/stl.h>

namespace om::ffi {

namespace {
bool _register_backend_group(
    const std::string& backend_name,
    const std::string& grp_name,
    tvm::ffi::Variant<
        tvm::ffi::TypedFunction<bool()>,
        tvm::ffi::TypedFunction<void()>> callback) {
  if (auto item = callback.as<tvm::ffi::TypedFunction<bool()>>()) {
    auto f = [item]() {
      try {
        return item.value()();
      } catch (const std::exception& e) {
        SPDLOG_ERROR("register_backend_group failed, error: {}", e.what());
      }
      return false;
    };
    return om::register_backend_group(backend_name, grp_name, f);
  } else if (auto item = callback.as<tvm::ffi::TypedFunction<void()>>()) {
    auto f = [item]() {
      try {
        item.value()();
        return true;
      } catch (const std::exception& e) {
        SPDLOG_ERROR("register_backend_group failed, error: {}", e.what());
        return false;
      }
      return false;
    };
    return om::register_backend_group(backend_name, grp_name, f);
  }
  return false;
}
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def(
      "omniback.register_backend_group", _register_backend_group);

  om::ffi::atexit_register(om::clear_groups);
}
// TVM_FFI_DLL_EXPORT_TYPED_FUNC(example, example)

}