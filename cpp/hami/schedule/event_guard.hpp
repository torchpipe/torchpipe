#pragma once

#include "hami/builtin/basic_backends.hpp"
namespace hami {
class EventGuard : public DependencyV0 {
    void custom_forward_with_dep(const std::vector<dict>& input_output,
                                 Backend* dependency) override;
};

// TASK_REQUEST_KEYKEY
}  // namespace hami