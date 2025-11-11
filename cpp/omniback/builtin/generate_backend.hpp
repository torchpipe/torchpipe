#pragma once

#include "omniback/core/backend.hpp"
#include "omniback/core/reflect.h"

namespace omniback {

/**
 * @brief 用于代码生成的后端基类. 通过OMNI_GENERATE_BACKEND生成和注册新的类
 * OMNI_GENERATE_BACKEND(Z, "S_v0[B,C]", "a=b,c=S_v0[Ad,f],e=f" )
 * A=>A;A[B,C]=>C;A[B,C[D]] => D;
 */
class GenerateBackend : public Backend {
 private:
  void impl_init(
      const std::unordered_map<std::string, std::string>& config,
      const dict& kwargs) override;

  void impl_inject_dependency(Backend* dependency) override final {
    if (!proxy_backend_) {
      throw std::runtime_error("GenerateBackend was not initialized yet");
    } else
      proxy_backend_->inject_dependency(dependency);
  }

  void impl_forward_with_dep(
      const std::vector<dict>& inputs,
      Backend& dependency) override {
    proxy_backend_->forward_with_dep(inputs, dependency);
  }

  void impl_forward(const std::vector<dict>& inputs) override {
    proxy_backend_->forward(inputs);
  }
  [[nodiscard]] virtual size_t impl_max() const override {
    return proxy_backend_->max();
  }

  [[nodiscard]] virtual size_t impl_min() const override {
    return proxy_backend_->min();
  }

 protected:
  std::unique_ptr<Backend> proxy_backend_;

 private:
  virtual std::pair<std::string, std::string> order() const = 0;
  void replace_dependency_config(
      std::unordered_map<std::string, std::string>& config,
      std::string dst_key);
  std::string get_latest_backend(const std::string& setting) const;

  std::unordered_map<std::string, std::string> parse_order_config(
      const std::string& setting);
};
} // namespace omniback

// see also OMNI_PROXY
#define OMNI_GENERATE_BACKEND(ClsName, main_backend, config)     \
  class ClsName : public omniback::GenerateBackend {             \
   private:                                                      \
    std::pair<std::string, std::string> order() const override { \
      return {main_backend, config};                             \
    }                                                            \
  };                                                             \
  OMNI_REGISTER_BACKEND(ClsName);
