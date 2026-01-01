#include "omniback/builtin/launch.hpp"
#include "omniback/helper/base_logging.hpp"
#include "omniback/helper/macro.h"
#include "omniback/helper/string.hpp"

namespace omniback {

void LaunchBase::impl_init(
    const std::unordered_map<std::string, std::string>& config,
    const dict& kwargs) {
  auto name = backend::get_dependency_name(this, config, "LaunchBase");

  {
    Backend* remote_dependency = OMNI_INSTANCE_GET(Backend, name);
    OMNI_ASSERT(remote_dependency);

    inject_dependency(remote_dependency);
  }
  post_init(config, kwargs);
}

void Forward::impl_inject_dependency(Backend* dependency) {
  if (injected_dependency_) {
    throw std::runtime_error(
        "inject_dependency  should not be called twice by Forward");
  } else {
    injected_dependency_ = dependency;
  }
}

void LaunchBase::impl_inject_dependency(Backend* dependency) {
  if (dependency == nullptr) {
    throw std::invalid_argument("null dependency is not allowed");
  }
  if (injected_dependency_) {
    injected_dependency_->inject_dependency(dependency);
  } else {
    injected_dependency_ = dependency;
  }
}

OMNI_REGISTER(Backend, Launch);
OMNI_REGISTER(Backend, Init);
OMNI_REGISTER(Backend, Forward);

} // namespace omniback