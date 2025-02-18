#pragma once

#include "hami/builtin/basic_backends.hpp"
#include "hami/helper/string.hpp"

namespace hami {
class Proxy : public Backend {
 public:
  void init(const std::unordered_map<string, string>& config, const dict& dict_config) override {
    // proxy_backend_ = owned_backend_.get();
  }
  void inject_dependency(Backend* dependency) override final {
    if (!proxy_backend_) {
      proxy_backend_ = dependency;
    } else
      proxy_backend_->inject_dependency(dependency);
  }

  void forward(const std::vector<dict>& inputs, Backend* dependency) override {
    proxy_backend_->forward(inputs, dependency);
  }
  bool try_forward(const std::vector<dict>& input_output, size_t timeout) override {
    return proxy_backend_->try_forward(input_output, timeout);
  }
  void forward(const std::vector<dict>& inputs) override { proxy_backend_->forward(inputs); }
  [[nodiscard]] virtual size_t max() const override { return proxy_backend_->max(); }

  [[nodiscard]] virtual size_t min() const override { return proxy_backend_->min(); }

 protected:
  // Backend* dependency_{nullptr};
  Backend* proxy_backend_{nullptr};
  std::unique_ptr<Backend> owned_backend_;
};

class BackendProxy : public Proxy {
 public:
  void init(const std::unordered_map<std::string, std::string>& config,
            const dict& dict_config) override;

 private:
  // std::unique_ptr<Backend> owned_backend_;
};

#define HAMI_PROXY(derived_aspect_cls, dependency_setting)                                        \
  class derived_aspect_cls : public Proxy {                                                       \
   private:                                                                                       \
   public:                                                                                        \
    void init(const std::unordered_map<string, string>& config,                                   \
              const dict& dict_config) override final {                                           \
      auto new_conf = config;                                                                     \
      auto backend_config = str::flatten_brackets(dependency_setting);                            \
      HAMI_ASSERT(backend_config.size() == 2);                                                    \
      std::string old = std::string(#derived_aspect_cls) + "::dependency";                        \
      if (new_conf.find(old) != new_conf.end()) {                                                 \
        auto backend_names = str::items_split(backend_config.at(1), ',', '[', ']');               \
        HAMI_ASSERT(backend_names.size() >= 1, " Aspect: backend_names.size() should >= 1");      \
        std::string new_dep = backend_names.back() + "::dependency";                              \
        new_conf[new_dep] = new_conf[old];                                                        \
        new_conf.erase(old);                                                                      \
      }                                                                                           \
      auto principal = backend_config.at(0);                                                      \
      owned_backend_ = std::unique_ptr<Backend>(HAMI_CREATE(Backend, principal));                 \
      if (!owned_backend_) throw std::runtime_error(principal + " is not a valid backend name."); \
      new_conf[principal + "::dependency"] = backend_config.at(1);                                \
      owned_backend_->init(new_conf, dict_config);                                                \
      proxy_backend_ = owned_backend_.get();                                                      \
    }                                                                                             \
  };                                                                                              \
  HAMI_REGISTER(Backend, derived_aspect_cls);

}  // namespace hami
