#pragma once

#include <string>
#include <vector>

#include "omniback/builtin/basic_backends.hpp"

namespace omniback {

/**
 * @brief Or[A,B], if A has no result, then execute B
 *
 */
class Or final : public Container {
 public:
  virtual void post_init(
      const std::unordered_map<std::string, std::string>&,
      const dict&) override;

  virtual void impl_forward(const std::vector<dict>&) override;
};

} // namespace omniback