#pragma once
#include "hami/builtin/basic_backends.hpp"
namespace hami {

class EventGuard : public Dependency {
  void forward_impl(const std::vector<dict>& input_output, Backend* dependency) override;
};

// TASK_REQUEST_KEYKEY
}  // namespace hami