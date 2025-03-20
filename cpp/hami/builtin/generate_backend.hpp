#pragma once

#include "hami/core/backend.hpp"
#include "hami/core/reflect.h"

namespace hami {

/**
 * @brief 用于代码生成的后端基类. 通过HAMI_GENERATE_BACKEND生成和注册新的类
 * HAMI_GENERATE_BACKEND(Z, "S_v0[B,C]", "a=b,c=S_v0[Ad,f],e=f" )
 * A=>A;A[B,C]=>C;A[B,C[D]] => D;
 */
class GenerateBackend : public Backend {
 private:
  void impl_init(const std::unordered_map<std::string, std::string>& config,
                 const dict& kwargs) override;

  void impl_inject_dependency(Backend* dependency) override final {
    if (!proxy_backend_) {
      throw std::runtime_error("GenerateBackend was not initialized yet");
    } else
      proxy_backend_->inject_dependency(dependency);
  }

  void impl_forward_with_dep(const std::vector<dict>& inputs,
                             Backend* dependency) override {
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
  // Backend* dependency_{nullptr};
  // Backend* proxy_backend_{nullptr};
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
}  // namespace hami

// see also HAMI_PROXY
#define HAMI_GENERATE_BACKEND(ClsName, main_backend, config)     \
  class ClsName : public hami::GenerateBackend {                 \
   private:                                                      \
    std::pair<std::string, std::string> order() const override { \
      return {main_backend, config};                             \
    }                                                            \
  };                                                             \
  HAMI_REGISTER_BACKEND(ClsName);
