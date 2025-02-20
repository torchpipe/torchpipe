#include "hami/builtin/proxy.hpp"
#include "hami/core/helper.hpp"
#include "hami/core/reflect.h"

namespace hami {

void BackendProxy::init(const std::unordered_map<string, string>& config,
                        const dict& dict_config) {
    auto iter = config.find("backend");
    HAMI_ASSERT(iter != config.end(),
                "BackendProxy configuration error: 'backend' parameter is "
                "missing. Please ensure it "
                "is properly specified in the configuration." +
                    debug_node_info(config));
    auto new_config = config;
    std::string backend = iter->second;
    auto main_backend = str::brackets_split(backend, new_config);
    owned_backend_ =
        std::unique_ptr<Backend>(HAMI_CREATE(Backend, main_backend));
    HAMI_ASSERT(owned_backend_, "`" + backend + "` is not a valid backend");
    owned_backend_->init(new_config, dict_config);
    proxy_backend_ = owned_backend_.get();
}

void Placeholder::init(const std::unordered_map<string, string>& config,
                       const dict& dict_config) {
    auto name = HAMI_OBJECT_NAME(Backend, this);
    HAMI_ASSERT(name != std::nullopt);

    auto iter = config.find(*name + "::dependency");
    HAMI_ASSERT(iter != config.end(),
                "Register: Dependency configuration missing for " + *name);
    HAMI_INSTANCE_REGISTER(Backend, iter->second, this);
}

HAMI_REGISTER(Backend, BackendProxy);
HAMI_REGISTER(Backend, Placeholder);

}  // namespace hami