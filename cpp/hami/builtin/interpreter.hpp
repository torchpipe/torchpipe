#pragma once
#include "hami/core/backend.hpp"
#include "hami/builtin/proxy.hpp"
namespace hami {

constexpr auto DEFAULT_INIT_CONFIG =
    "List[RegisterInstances[BackgroundThread[BackendProxy]], RegisterNode[Aspect[Batching, "
    "InstanceDispatcher]]]";

class Interpreter : public Proxy {
 public:
  void init(const std::unordered_map<std::string, std::string>& config,
            const dict& dict_config) override final;

 private:
  std::vector<std::unique_ptr<Backend>> inited_dependencies_;
};
}  // namespace hami