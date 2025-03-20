#pragma once

#include <memory>

#include "hami/builtin/basic_backends.hpp"
#include "hami/helper/macro.h"
#include "hami/helper/params.hpp"
namespace hami {

class Condition : public DependencyV0 {
 private:
  std::function<bool(const dict&)> condition_;

 public:
  void pre_init(const std::unordered_map<std::string, std::string>& config,
                const dict& kwargs) override final;
  virtual void init_dep_impl(
      const std::unordered_map<std::string, std::string>& config,
      const dict& kwargs) {}
  void custom_forward_with_dep(const std::vector<dict>& inputs,
                               Backend* dependency) override final {
    std::vector<dict> valid_inputs;
    for (auto& input : inputs) {
      if (condition_(input)) {
        valid_inputs.push_back(input);
      }
    }
    if (!valid_inputs.empty()) {
      dependency->safe_forward(valid_inputs);
    }
  }

  virtual std::function<bool(const dict&)> condition_impl() const = 0;

 public:
  // std::function<bool(const dict&)> condition_impl() const override {
  //   return [this](const dict& input) { return input->find(key_) !=
  //   input->end(); };
  // }
};
}  // namespace hami