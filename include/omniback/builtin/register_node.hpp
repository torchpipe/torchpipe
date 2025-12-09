#pragma once

#include "basic_backends.hpp"
#include "omniback/core/backend.hpp"

namespace omniback {
class Register : public Dependency {
 public:
  void impl_init(
      const std::unordered_map<std::string, std::string>& config,
      const dict& kwargs) override final;
  // void impl_forward_with_dep(const std::vector<dict>& ios, Backend& dep) {
  //   dep.forward(ios);
  // }

 private:
  std::unique_ptr<Backend> owned_backend_;
};
} // namespace omniback