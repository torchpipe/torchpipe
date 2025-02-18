#pragma once

#include <memory>
#include "hami/builtin/basic_backends.hpp"
#include "hami/helper/params.hpp"
#include "hami/helper/macro.h"
#include "hami/core/task_keys.hpp"

namespace hami {

class LaunchBase : public Backend {
  void inject_dependency(Backend* dependency) override final;

  virtual void forward(const std::vector<dict>& input_output) override final {
    forward(input_output, injected_dependency_);
  }
  virtual void forward(const std::vector<dict>& input_output, Backend* dependency) override final {
    if (dependency == nullptr) {
      throw std::invalid_argument("null dependency is not allowed");
    }
    forward_impl(input_output, dependency);
  }

  [[nodiscard]] size_t max() const override {
    return injected_dependency_ ? injected_dependency_->max() : std::numeric_limits<size_t>::max();
  }
  [[nodiscard]] size_t min() const override {
    return injected_dependency_ ? injected_dependency_->min() : 1;
  }

  /**
   * @brief Optional support for A[B], where A::dependency=B is passed as a parameter and
   * injected as a dependency.
   *
   * The dependency instance is registered as {node_name}.{A}.{B}.
   */
  void init(const std::unordered_map<std::string, std::string>& config,
            const dict& dict_config) override final;

  virtual void post_init(const std::unordered_map<std::string, std::string>& config,
                         const dict& dict_config) {}

 private:
  virtual void forward_impl(const std::vector<dict>& input_output, Backend* dependency) = 0;

 protected:
  Backend* injected_dependency_{nullptr};  ///< The dependency.
};

class Init : public LaunchBase {
 public:
  void post_init(const std::unordered_map<std::string, std::string>& config,
                 const dict& dict_config) override final {
    injected_dependency_->init(config, dict_config);
  }
  void forward_impl(const std::vector<dict>& input_output, Backend* dependency) override final {
    for (auto& input : input_output) {
      (*input)[TASK_RESULT_KEY] = input->at(TASK_DATA_KEY);
    }
  }
};

class Forward : public LaunchBase {
 public:
  void forward_impl(const std::vector<dict>& input_output, Backend* dependency) override final {
    dependency->safe_forward(input_output);
  }
};

class Launch : public LaunchBase {
 public:
  void post_init(const std::unordered_map<std::string, std::string>& config,
                 const dict& dict_config) override final {
    injected_dependency_->init(config, dict_config);
  }
  void forward_impl(const std::vector<dict>& input_output, Backend* dependency) override final {
    dependency->safe_forward(input_output);
  }
};

}  // namespace hami