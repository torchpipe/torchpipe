#pragma once

#include "hami/core/backend.hpp"
#include "basic_backends.hpp"

namespace hami {
class Register : public DependencyV0 {
 public:
  void pre_init(const std::unordered_map<std::string, std::string>& config,
                const dict& kwargs) override final;
};
}  // namespace hami