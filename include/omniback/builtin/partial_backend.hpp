#pragma once
#include <memory>
#include <functional>
#include <vector>
#include "omniback/core/backend.hpp"
#include <tvm/ffi/function.h>
#include <tvm/ffi/container/map.h>
#include <tvm/ffi/container/array.h>
// #include <tvm/ffi/container/variant.h>

#include "omniback/ffi/dict.h"

namespace om {
// void OMNI_PARTIAL_REGISTER(){

// }

using FfiPartialInitFunc = tvm::ffi::TypedFunction<void(
    om::ffi::DictRef,
    const tvm::ffi::Map<tvm::ffi::String, tvm::ffi::String>&,
    tvm::ffi::Optional<om::ffi::DictRef>)>;

using FfiPartialForwardFunc = tvm::ffi::TypedFunction<void(om::ffi::DictRef, const tvm::ffi::Array<om::ffi::DictRef>&)>;
// using FfiPartialMaxMinFunc = tvm::ffi::Variant<tvm::ffi::TypedFunction<uint32_t(om::ffi::DictRef)>, uint32_t>;
using FfiPartialMaxMinFunc = tvm::ffi::TypedFunction<uint32_t(om::ffi::DictRef)>;

class OMNI_EXPORT PartialBackend : public Backend {
 public:
  using InitCallback = std::optional<std::function<void(
      const om::dict&,
      const std::unordered_map<std::string, std::string>&,
      const om::dict&)>>;
  using ForwardCallback =
      std::optional<std::function<void(
        const om::dict&,
        const std::vector<om::dict>&)>>;
  using MaxMinCallback = std::function<uint32_t(const om::dict&)>;
  
  static InitCallback partial_ffi2init(FfiPartialInitFunc init_func);
  static ForwardCallback partial_ffi2forward(FfiPartialForwardFunc forward_func);
  static MaxMinCallback partial_ffi2maxmin(FfiPartialMaxMinFunc max_min_func);

  PartialBackend(
      std::string name,
      std::string grp = "om",
      uint32_t force_max = 0 /**, bool lazy_setting = false */)
      : name_(name) {
    std::string prefix = grp +"."+ name;
    data_ = om::make_dict();
    auto init_func = tvm::ffi::Function::GetGlobal(prefix +".init");
    if (init_func)
      init_cb_ = partial_ffi2init(
          init_func.value());
    auto forward_func =
        tvm::ffi::Function::GetGlobal(prefix + ".forward");
    if (forward_func)
      forward_cb_ = partial_ffi2forward(
          forward_func.value());
    if (force_max != 0){
      max_cb_ = [force_max](const om::dict&) { return force_max; };
    }else
    {
      auto max_func = tvm::ffi::Function::GetGlobal(prefix + ".max");
      if (max_func)
        max_cb_ = partial_ffi2maxmin(
            max_func.value());
      else{
        max_cb_ = [](const om::dict&) { return Backend::default_max(); };
      }
    }
    auto min_func = tvm::ffi::Function::GetGlobal(prefix + ".min");
    if (min_func)
      min_cb_ = partial_ffi2maxmin(
          min_func.value());
      else{
        min_cb_ = [](const om::dict&) { return Backend::default_min(); };
      }
  }

 private:
  void impl_init(
      const std::unordered_map<std::string, std::string>& params,
      const om::dict& options) override {
    if (init_cb_)
      {
        init_cb_.value()(data_, params, options);
      }
  }

  void impl_forward(const std::vector<om::dict>& ios) override {
    if (forward_cb_)
      {
        forward_cb_.value()(data_, ios);
      }
  }

  uint32_t impl_max() const override {
    return max_cb_(data_);
  }

  uint32_t impl_min() const override {
    return min_cb_(data_);
  }

  InitCallback init_cb_;
  ForwardCallback forward_cb_;
  MaxMinCallback max_cb_;
  MaxMinCallback min_cb_;
  std::string name_;
  om::dict data_;
};


} // namespace om