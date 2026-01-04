#include "omniback/builtin/callback_backend.hpp"
#include "omniback/py/pybackend.hpp"

namespace omniback::py {
using omniback::CallbackBackend ;
std::unique_ptr<omniback::Backend> object2backend(
    SelfType py_obj,
    tvm::ffi::Optional<tvm::ffi::TypedFunction<void(
        SelfType,
        const tvm::ffi::Map<tvm::ffi::String, tvm::ffi::String>&,
        tvm::ffi::Optional<PyDictRef>)>> init_func,
    tvm::ffi::Optional<
        tvm::ffi::TypedFunction<void(SelfType, tvm::ffi::Array<PyDictRef>)>>
        forward_func,
    tvm::ffi::Optional<tvm::ffi::Variant<
        tvm::ffi::TypedFunction<uint32_t(SelfType)>,
        uint32_t>> max_func,
    tvm::ffi::Optional<tvm::ffi::Variant<
        tvm::ffi::TypedFunction<uint32_t(SelfType)>,
        uint32_t>> min_func) {
  CallbackBackend::InitCallback init_cb;
  CallbackBackend::ForwardCallback forward_cb;
  CallbackBackend::MaxCallback max_cb;
  CallbackBackend::MinCallback min_cb;
  
  if (init_func.has_value()) {
    init_cb = 
        [py_obj, init_func](
            const std::unordered_map<std::string, std::string>& params,
            const omniback::dict& options) {

          tvm::ffi::Map<tvm::ffi::String, tvm::ffi::String> params_map{params.begin(), params.end()};

          tvm::ffi::Optional<PyDictRef> options_dict;
          if (options) {
            auto options_dict =
                tvm::ffi::make_object<omniback::ffi::DictObj>(options);
          }

          (init_func.value())(py_obj, params_map, options_dict);
        };
    };

    if (forward_func.has_value()) {
        forward_cb = [py_obj, forward_func](const std::vector<omniback::dict>& ios) {

          tvm::ffi::Array<PyDictRef> arr;
          for (const auto& io_dict : ios) {
            auto dict_obj =
                tvm::ffi::make_object<omniback::ffi::DictObj>(io_dict);
            arr.push_back(PyDictRef(dict_obj));
          }

          (forward_func.value())(py_obj, std::move(arr));
        };
    }

    if (max_func.has_value()) {
      const auto& max_val = max_func.value();

      if (auto value = max_val.as<uint32_t>()) {
        uint32_t max_result = value.value();
        max_cb = [max_result]() { return max_result; };
      } else if (
          auto func =
              max_val.as<tvm::ffi::TypedFunction<uint32_t(SelfType)>>()) {
        max_cb = [py_obj, func]() {
          return (func.value())(py_obj);
        };
      }
    }

    if (min_func.has_value()) {
      const auto& min_val = min_func.value();

      if (auto value = min_val.as<uint32_t>()) {
        uint32_t min_result = value.value();
        min_cb = [min_result]() { return min_result; };
      } else if (
          auto func =
              min_val.as<tvm::ffi::TypedFunction<uint32_t(SelfType)>>()) {
        min_cb = [py_obj, func]() { return (func.value())(py_obj); };
      }
    }

  return std::make_unique<CallbackBackend>(init_cb, forward_cb, max_cb, min_cb);
};
}
