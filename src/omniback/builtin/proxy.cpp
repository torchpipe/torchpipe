#include "omniback/builtin/proxy.hpp"
#include "omniback/core/helper.hpp"
#include "omniback/core/reflect.h"
#include "omniback/helper/base_logging.hpp"

namespace omniback {

void BackendProxy::impl_init(
    const std::unordered_map<string, string>& config,
    const dict& kwargs) {
  auto iter = config.find("backend");
  OMNI_ASSERT(
      iter != config.end(),
      "BackendProxy configuration error: 'backend' parameter is "
      "missing. Please ensure it "
      "is properly specified in the configuration." +
          debug_node_info(config));
  auto new_config = config;
  std::string backend = iter->second;
  auto main_backend = str::brackets_split(backend, new_config);
  owned_backend_ = std::unique_ptr<Backend>(OMNI_CREATE(Backend, main_backend));
  OMNI_ASSERT(owned_backend_, "`" + backend + "` is not a valid backend");
  owned_backend_->init(new_config, kwargs);
  proxy_backend_ = owned_backend_.get();
}

void Placeholder::impl_init(
    const std::unordered_map<string, string>& config,
    const dict& kwargs) {
  auto name = OMNI_OBJECT_NAME(Backend, this);
  OMNI_ASSERT(name != std::nullopt);

  auto iter = config.find(*name + "::dependency");
  OMNI_ASSERT(
      iter != config.end(),
      "Register: Dependency configuration missing for " + *name);
  OMNI_INSTANCE_REGISTER(Backend, iter->second, this);
}

OMNI_REGISTER(Backend, BackendProxy);
OMNI_REGISTER(Backend, Placeholder);

void Reflect::impl_init(
    const std::unordered_map<string, string>& in_config,
    const dict& kwargs) {
  auto config = in_config;
  std::string default_dep;

  default_dep =
      parse_dependency_from_param(this, config, "backend", "Identity");
  SPDLOG_INFO("reflect default_dep = {}", default_dep);
  owned_backend_ = init_backend(default_dep, config, kwargs);

  OMNI_ASSERT(owned_backend_);
  // owned_backend_->init(config, kwargs);
  proxy_backend_ = owned_backend_.get();
}

OMNI_REGISTER(Backend, Reflect, "Reflect,ProxyFromParam");

void Proxy::impl_init(
    const std::unordered_map<string, string>& config,
    const dict& kwargs) {
  OMNI_ASSERT(proxy_backend_ == nullptr && owned_backend_ == nullptr, "Proxy: already initialized");
  auto name = backend::get_dependency_name(this, config, "Proxy");

  proxy_backend_ = OMNI_INSTANCE_GET(Backend, name);
  OMNI_ASSERT(proxy_backend_, "Proxy: backend not found : " + name);
  proxy_backend_->init(config, kwargs);
}

OMNI_REGISTER(Backend, Proxy, "Proxy");

void DI_v0::impl_init(
    const std::unordered_map<string, string>& config,
    const dict& kwargs) {
  auto name = backend::get_dependency_name(this, config);
  auto re = str::str_split(name, ',');
  OMNI_ASSERT(
      re.size() == 2, "Usage: DI_v0[src_instance_name,target_instance_name]");
  proxy_backend_ = OMNI_INSTANCE_GET(Backend, re[0]);
  auto* dep = OMNI_INSTANCE_GET(Backend, re[1]);
  OMNI_ASSERT(
      proxy_backend_ && dep,
      "DI: instance name not found. Usage: DI_v0[ins1, ins2], here "
      "ins1 is a name of instance, not a name of class; name = " +
          re[0] + "/" + re[1]);
  proxy_backend_->inject_dependency(dep);
  SPDLOG_INFO(
      "DI: {} -> {} [{}, {}]",
      re[0],
      re[1],
      proxy_backend_->min(),
      proxy_backend_->max());
}

void DI::impl_init(
    const std::unordered_map<string, string>& config,
    const dict& kwargs) {
  auto name = backend::get_dependency_name(this, config);
  auto re = str::str_split(name, ',');
  OMNI_ASSERT(
      re.size() >= 2, "Usage: DI[instance_0,instance_1, ... , instance_n]");
  // proxy_backend_ = OMNI_INSTANCE_GET(Backend, re[0]);
  std::vector<Backend*> backends;
  for (size_t i = 0; i < re.size(); ++i) {
    auto* dep = OMNI_INSTANCE_GET(Backend, re[i]);
    OMNI_ASSERT(
        dep,
        "Usage: DI[instance_0,instance_1, ... , instance_n]. "
        "instance name = " +
            std::string(re[i]));
    backends.push_back(dep);
  }
  for (size_t i = re.size() - 2;; --i) {
    backends[i]->inject_dependency(backends[i + 1]);
    if (i == 0)
      break;
  }
  proxy_backend_ = backends[0];

  SPDLOG_INFO("DI: ", proxy_backend_->min(), proxy_backend_->max());
}

OMNI_REGISTER_BACKEND(DI_v0);
OMNI_REGISTER_BACKEND(DI);

} // namespace omniback
