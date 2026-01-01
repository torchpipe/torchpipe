
#pragma once

#include "omniback/builtin/proxy.hpp"
#include "omniback/core/backend.hpp"
namespace omniback {

// constexpr auto DEFAULT_INIT_CONFIG =
//     "List[InstancesRegister[BackgroundThread[BackendProxy]], "
//     "Register[Aspect[Batching, "
//     "InstanceDispatcher]]]";
constexpr auto DEFAULT_NODE_CONFIG =
    u8"IoCV0[SharedInstancesState,InstanceDispatcher,Batching;DI_v0[Batching,"
    "InstanceDispatcher]]";
// constexpr auto DEFAULT_NODE_CONFIG = "IoCV0[Pass;Identity]";
constexpr auto DEFAULT_INSTANCES_CONFIG =
    u8"BackgroundThread[Reflect[backend]]";
static const auto DEFAULT_INIT_CONFIG = std::string("List[InstancesRegister[") +
    DEFAULT_INSTANCES_CONFIG + "], " + "Register[" + DEFAULT_NODE_CONFIG + "]]";

class Interpreter : public Proxy {
 private:
  void impl_init(
      const std::unordered_map<std::string, std::string>& config,
      const dict& kwargs) override final;

 private:
  std::vector<std::unique_ptr<Backend>> inited_dependencies_;
};
} // namespace omniback