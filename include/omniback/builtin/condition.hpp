#pragma once

#include <memory>

#include "omniback/builtin/basic_backends.hpp"
#include "omniback/helper/macro.h"
#include "omniback/helper/params.hpp"
namespace omniback {

class Condition : public DependencyV0 {
 private:
  std::function<bool(const dict&)> condition_;

 public:
  void pre_init(
      const std::unordered_map<std::string, std::string>& config,
      const dict& kwargs) override final;
  virtual void init_dep_impl(
      const std::unordered_map<std::string, std::string>& config,
      const dict& kwargs) {}
  void custom_forward_with_dep(
      const std::vector<dict>& inputs,
      Backend& dependency) override final {
    std::vector<dict> valid_inputs;
    for (auto& input : inputs) {
      if (condition_(input)) {
        valid_inputs.push_back(input);
      }
    }
    if (!valid_inputs.empty()) {
      dependency.safe_forward(valid_inputs);
    }
  }

  virtual std::function<bool(const dict&)> condition_impl() const = 0;

 public:
  // std::function<bool(const dict&)> condition_impl() const override {
  //   return [this](const dict& input) { return input->find(key_) !=
  //   input->end(); };
  // }
};

class NotHasKey : public BackendOne {
  void impl_init(
      const std::unordered_map<std::string, std::string>& params,
      const dict& options) override;
  void forward(const dict& io) override;

 private:
  std::string key_;
  std::unique_ptr<Backend> dependency_{nullptr};
};
class HasKey : public BackendOne {
  void impl_init(
      const std::unordered_map<std::string, std::string>& params,
      const dict& options) override;
  void forward(const dict& io) override;

 private:
  std::string key_;
  std::unique_ptr<Backend> dependency_a_{nullptr};
  std::unique_ptr<Backend> dependency_b_{nullptr};
};
} // namespace omniback