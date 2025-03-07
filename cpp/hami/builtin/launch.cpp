#include "hami/builtin/launch.hpp"
#include "hami/helper/string.hpp"
#include "hami/helper/base_logging.hpp"
#include "hami/helper/macro.h"

namespace hami {

void LaunchBase::init(
    const std::unordered_map<std::string, std::string>& config,
    const dict& dict_config) {
    auto name = backend::get_dependency_name(this, config, "LaunchBase");

    {
        Backend* remote_dependency = HAMI_INSTANCE_GET(Backend, name);
        HAMI_ASSERT(remote_dependency);

        inject_dependency(remote_dependency);
    }
    post_init(config, dict_config);
}

void Forward::inject_dependency(Backend* dependency) {
    if (injected_dependency_) {
        throw std::runtime_error(
            "inject_dependency  should not be called twice by Forward");
    } else {
        injected_dependency_ = dependency;
    }
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