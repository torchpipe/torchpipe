#pragma once
#include <memory>
#include <functional>
#include <vector>
#include "omniback/core/backend.hpp"

namespace omniback {

class CallbackBackend : public Backend {
 public:
  using InitCallback = std::function<void(
      const std::unordered_map<std::string, std::string>&,
      const omniback::dict&)>;
  using ForwardCallback =
      std::function<void(const std::vector<omniback::dict>&)>;
  using MaxCallback = std::function<uint32_t()>;
  using MinCallback = std::function<uint32_t()>;

  CallbackBackend(
      InitCallback init_cb,
      ForwardCallback forward_cb,
      MaxCallback max_cb,
      MinCallback min_cb)
      : init_cb_(std::move(init_cb)),
        forward_cb_(std::move(forward_cb)),
        max_cb_(std::move(max_cb)),
        min_cb_(std::move(min_cb)) {
    if (!max_cb_) {
      max_cb_ = []() { return Backend::default_max(); };
    }
    if (!min_cb_) {
      min_cb_ = []() { return Backend::default_min(); };
    }
  }

 private:
  void impl_init(
      const std::unordered_map<std::string, std::string>& params,
      const omniback::dict& options) override {
    if (init_cb_)
      init_cb_(params, options);
  }

  void impl_forward(const std::vector<omniback::dict>& ios) override {
    if (forward_cb_)
      forward_cb_(ios);
  }

  uint32_t impl_max() const override {
    return max_cb_();
  }

  uint32_t impl_min() const override {
    return min_cb_();
  }

  InitCallback init_cb_;
  ForwardCallback forward_cb_;
  MaxCallback max_cb_;
  MinCallback min_cb_;
};

} // namespace omniback