#pragma once

#include "basic_backends.hpp"
#include "hami/core/backend.hpp"

namespace hami {
class Register : public Dependency {
 public:
  void impl_init(
      const std::unordered_map<std::string, std::string>& config,
      const dict& kwargs) override final;
  void impl_forward_with_dep(
      const std::vector<dict>& input_output,
      Backend* dep) {
    dep->forward(input_output);
  }

 private:
  std::unique_ptr<Backend> owned_backend_;
};
} // namespace hami