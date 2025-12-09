#pragma once

#include "omniback/builtin/basic_backends.hpp"
namespace omniback {
class EventGuard : public DependencyV0 {
  void custom_forward_with_dep(
      const std::vector<dict>& ios,
      Backend& dependency) override;
};

// TASK_REQUEST_KEYKEY
} // namespace omniback