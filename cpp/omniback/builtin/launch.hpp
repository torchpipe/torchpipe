#pragma once

#include <memory>
#include "omniback/builtin/basic_backends.hpp"
#include "omniback/core/task_keys.hpp"
#include "omniback/helper/macro.h"
#include "omniback/helper/params.hpp"

namespace omniback {

class LaunchBase : public Backend {
  void impl_inject_dependency(Backend* dependency) override;

  virtual void impl_forward(
      const std::vector<dict>& input_output) override final {
    forward_with_dep(input_output, *injected_dependency_);
  }
  virtual void impl_forward_with_dep(
      const std::vector<dict>& input_output,
      Backend& dependency) override final {
    custom_forward_with_dep(input_output, dependency);
  }

  [[nodiscard]] size_t impl_max() const override {
    return injected_dependency_ ? injected_dependency_->max()
                                : std::numeric_limits<size_t>::max();
  }
  [[nodiscard]] size_t impl_min() const override {
    return injected_dependency_ ? injected_dependency_->min() : 1;
  }

  /**
   * @brief Optional support for A[B], where A::dependency=B is passed as a
   * parameter and injected as a dependency.
   *
   * The dependency instance is registered as {node_name}.{A}.{B}.
   */
  void impl_init(
      const std::unordered_map<std::string, std::string>& config,
      const dict& kwargs) override final;

 public:
  virtual void post_init(
      const std::unordered_map<std::string, std::string>& config,
      const dict& kwargs) {}

 private:
  virtual void custom_forward_with_dep(
      const std::vector<dict>& input_output,
      Backend& dependency) = 0;

 protected:
  Backend* injected_dependency_{nullptr}; ///< The dependency.
};

class Init : public LaunchBase {
 public:
  void post_init(
      const std::unordered_map<std::string, std::string>& config,
      const dict& kwargs) override final {
    injected_dependency_->init(config, kwargs);
  }
  void custom_forward_with_dep(
      const std::vector<dict>& input_output,
      Backend& dependency) override final {
    for (auto& input : input_output) {
      (*input)[TASK_RESULT_KEY] = input->at(TASK_DATA_KEY);
    }
  }
};

class Forward : public LaunchBase {
 public:
  void custom_forward_with_dep(
      const std::vector<dict>& input_output,
      Backend& dependency) override final {
    dependency.safe_forward(input_output);
  }
  void impl_inject_dependency(Backend* dependency) override final;

  [[nodiscard]] size_t impl_max() const override {
    return injected_dependency_ ? injected_dependency_->max()
                                : std::numeric_limits<size_t>::max();
  }
  [[nodiscard]] size_t impl_min() const override {
    return injected_dependency_ ? injected_dependency_->min() : 1;
  }
};

class Launch : public LaunchBase {
 public:
  void post_init(
      const std::unordered_map<std::string, std::string>& config,
      const dict& kwargs) override final {
    injected_dependency_->init(config, kwargs);
  }
  void custom_forward_with_dep(
      const std::vector<dict>& input_output,
      Backend& dependency) override final {
    dependency.safe_forward(input_output);
  }
};

} // namespace omniback