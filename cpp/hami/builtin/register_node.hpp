#pragma once

#include "hami/core/backend.hpp"
#include "basic_backends.hpp"

namespace hami {
class RegisterNode : public Dependency {
 public:
  void pre_init(const std::unordered_map<std::string, std::string>& config,
                const dict& dict_config) override final;
};
}  // namespace hami