#include "hami/builtin/proxy.hpp"
#include "hami/core/helper.hpp"
#include "hami/core/reflect.h"
#include "hami/helper/base_logging.hpp"

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

void Reflect::init(const std::unordered_map<string, string>& config,
                   const dict& dict_config) {
    constexpr auto default_name = "Reflect";
    auto name = HAMI_OBJECT_NAME(Backend, this);
    if (name == std::nullopt) {
        name = default_name;
        SPDLOG_WARN(
            "{}::init, it seems this instance was not created via "
            "reflection, using default name {}. "
            "Please configure its dependency via the parameter "
            "{}::dependency",
            *name, *name, *name);
    }
    auto iter = config.find(*name + "::dependency");
    HAMI_ASSERT(
        iter != config.end(),
        *name + "::dependency not found. Call this backend through A[B].");
    iter = config.find(iter->second);
    HAMI_ASSERT(iter != config.end(),
                "configuration missing for " + iter->second);

    // owned_backend_ =
    //     std::unique_ptr<Backend>(HAMI_CREATE(Backend, iter->second));
    owned_backend_ = init_backend(iter->second, config, dict_config);
    // HAMI_ASSERT(owned_backend_);
    // owned_backend_->init(config, dict_config);
    proxy_backend_ = owned_backend_.get();
}

HAMI_REGISTER(Backend, Reflect, "Reflect,ProxyFromParam");

void Proxy::init(const std::unordered_map<string, string>& config,
                 const dict& dict_config) {
    auto name = backend::get_dependency_name(this, config, "Proxy");

    proxy_backend_ = HAMI_INSTANCE_GET(Backend, name);
    HAMI_ASSERT(proxy_backend_, "Proxy: backend not found : " + name);
}

HAMI_REGISTER(Backend, Proxy, "Proxy");

void DI::init(const std::unordered_map<string, string>& config,
              const dict& dict_config) {
    auto name = backend::get_dependency_name(this, config);
    auto re = str::str_split(name, ',');
    HAMI_ASSERT(re.size() == 2,
                "Usage: DI[src_instance_name,target_instance_name]");
    proxy_backend_ = HAMI_INSTANCE_GET(Backend, re[0]);
    auto* dep = HAMI_INSTANCE_GET(Backend, re[1]);
    HAMI_ASSERT(proxy_backend_ && dep,
                "DI: backend not found. name = " + re[0] + re[1]);
    proxy_backend_->inject_dependency(dep);
}

HAMI_REGISTER(Backend, DI, "DI");

}  // namespace hami
