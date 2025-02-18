#include "hami/builtin/launch.hpp"
#include "hami/helper/string.hpp"
#include "hami/helper/base_logging.hpp"
#include "hami/helper/macro.h"

namespace hami {

void LaunchBase::init(const std::unordered_map<std::string, std::string>& config,
                      const dict& dict_config) {
  constexpr auto default_name = "LaunchBase";
  auto name = HAMI_OBJECT_NAME(Backend, this);
  if (name == std::nullopt) {
    name = default_name;
    SPDLOG_WARN(
        "{}::init, it seems this instance was not created via reflection, using default name {}. "
        "Please configure its dependency via the parameter {}::dependency",
        *name, *name, *name);
  }
  auto iter = config.find(*name + "::dependency");
  HAMI_ASSERT(iter != config.end());
  {
    Backend* remote_dependency = HAMI_INSTANCE_GET(Backend, iter->second);
    HAMI_ASSERT(remote_dependency);

    inject_dependency(remote_dependency);
  }
  post_init(config, dict_config);
}

void LaunchBase::inject_dependency(Backend* dependency) {
  if (dependency == nullptr) {
    throw std::invalid_argument("null dependency is not allowed");
  }
  if (injected_dependency_) {
    injected_dependency_->inject_dependency(dependency);
  } else {
    injected_dependency_ = dependency;
  }
}

HAMI_REGISTER(Backend, Launch);
HAMI_REGISTER(Backend, Init);
HAMI_REGISTER(Backend, Forward);

}  // namespace hami