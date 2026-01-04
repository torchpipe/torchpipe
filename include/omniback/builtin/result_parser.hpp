#pragma once

#include <memory>
#include "omniback/builtin/basic_backends.hpp"
#include "omniback/helper/macro.h"
#include "omniback/helper/params.hpp"
namespace omniback {

class ResultParser : public DependencyV0 {
 private:
  std::function<void(const dict&)> parser_;

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
    dependency.safe_forward(inputs);
    for (const auto& item : inputs) {
      parser_(item);
    }
  }

  virtual std::function<void(const dict&)> parser_impl() const = 0;
};

} // namespace omniback