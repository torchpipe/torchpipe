// Copyright 2021-2025 NetEase.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include "omniback/builtin/basic_backends.hpp"
#include "omniback/core/reflect.h"
#include "omniback/helper/macro.h"
#include "omniback/helper/string.hpp"

namespace omniback {
class Proxy : public Backend {
 private:
  void impl_init(
      const std::unordered_map<string, string>& config,
      const dict& kwargs) override;

  void impl_inject_dependency(Backend* dependency) override {
    if (!proxy_backend_) {
      proxy_backend_ = dependency;
    } else
      proxy_backend_->inject_dependency(dependency);
  }

  void impl_forward_with_dep(const std::vector<dict>& ios, Backend& dependency)
      override {
    proxy_backend_->forward_with_dep(ios, dependency);
  }

  void impl_forward(const std::vector<dict>& ios) override {
    proxy_backend_->forward(ios);
  }
  [[nodiscard]] virtual size_t impl_max() const override {
    return proxy_backend_->max();
  }

  [[nodiscard]] virtual size_t impl_min() const override {
    return proxy_backend_->min();
  }

 protected:
  Backend* proxy_backend_{nullptr};
  std::unique_ptr<Backend> owned_backend_;
};

// class InstanceProxy : public Proxy {
//    public:
//     void impl_init(const std::unordered_map<string, string>& config,
//               const dict& kwargs) override;
// };

class DI_v0 : public Proxy {
 private:
  void impl_init(
      const std::unordered_map<string, string>& config,
      const dict& kwargs) override;
  void impl_inject_dependency(Backend* dependency) override {
    throw std::runtime_error("DI: inject_dependency Unillegal");
  }
};

class DI : public Proxy {
 private:
  void impl_init(
      const std::unordered_map<string, string>& config,
      const dict& kwargs) override;
  void impl_inject_dependency(Backend* dependency) override {
    throw std::runtime_error("DI: inject_dependency Unillegal");
  }
};
class Placeholder : public Proxy {
 private:
  void impl_init(
      const std::unordered_map<string, string>& config,
      const dict& kwargs) override;
  void impl_inject_dependency(Backend* dependency) override {
    if (!proxy_backend_) {
      proxy_backend_ = dependency;
    } else {
      throw std::runtime_error("Placeholder: inject_dependency called twice");
    }
    OMNI_ASSERT(!owned_backend_);
  }
};

class ProxyV2 : public Backend {
 public:
  std::pair<std::string, str::str_map> make_order(
      const std::string& setting,
      const str::str_map& dict_setting = {}) const {
    return {setting, dict_setting};
  }

  void impl_init(
      const std::unordered_map<string, string>& config,
      const dict& kwargs) override final {
    auto execorder = get_order();
    proxy_backend_ = init_backend(execorder.first, execorder.second);
  }
  virtual std::pair<std::string, str::str_map> get_order() const = 0;
  void impl_inject_dependency(Backend* dependency) override final {
    if (!proxy_backend_) {
      throw std::runtime_error("ProxyV2 was not initialized yet");
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
  // Backend* dependency_{nullptr};
  // Backend* proxy_backend_{nullptr};
  std::unique_ptr<Backend> proxy_backend_;
};

class BackendProxy : public Proxy {
 private:
  void impl_init(
      const std::unordered_map<std::string, std::string>& config,
      const dict& kwargs) override;

 private:
  // std::unique_ptr<Backend> owned_backend_;
};

class Reflect : public Proxy {
  void impl_init(
      const std::unordered_map<std::string, std::string>& config,
      const dict& kwargs) override;
};

#define OMNI_PROXY(derived_aspect_cls, setting, ...)                  \
  class derived_aspect_cls : public ProxyV2 {                         \
   public:                                                            \
    std::pair<std::string, str::str_map> get_order() const override { \
      return make_order(setting, ##__VA_ARGS__);                      \
    }                                                                 \
  };                                                                  \
  OMNI_REGISTER(Backend, derived_aspect_cls);

#define OMNI_PROXY_WITH_DEPENDENCY(derived_aspect_cls, dependency_setting)     \
  class derived_aspect_cls : public Proxy {                                    \
   private:                                                                    \
    void impl_init(                                                            \
        const std::unordered_map<string, string>& config,                      \
        const dict& kwargs) override final {                                   \
      auto new_conf = config;                                                  \
      auto backend_config = str::flatten_brackets(dependency_setting);         \
      OMNI_ASSERT(backend_config.size() == 2);                                 \
      std::string old = std::string(#derived_aspect_cls) + "::dependency";     \
      if (new_conf.find(old) != new_conf.end()) {                              \
        auto backend_names =                                                   \
            str::items_split(backend_config.at(1), ',', '[', ']');             \
        OMNI_ASSERT(                                                           \
            backend_names.size() >= 1,                                         \
            " Aspect: backend_names.size() should >= 1");                      \
        std::string new_dep = backend_names.back() + "::dependency";           \
        new_conf[new_dep] = new_conf[old];                                     \
        new_conf.erase(old);                                                   \
      }                                                                        \
      auto principal = backend_config.at(0);                                   \
      owned_backend_ =                                                         \
          std::unique_ptr<Backend>(OMNI_CREATE(Backend, principal));           \
      if (!owned_backend_)                                                     \
        throw std::runtime_error(principal + " is not a valid backend name."); \
      new_conf[principal + "::dependency"] = backend_config.at(1);             \
      owned_backend_->init(new_conf, kwargs);                                  \
      proxy_backend_ = owned_backend_.get();                                   \
    }                                                                          \
  };                                                                           \
  OMNI_REGISTER(Backend, derived_aspect_cls);

} // namespace omniback
