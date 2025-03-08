
#pragma once

#include "hami/core/backend.hpp"
#include "hami/builtin/proxy.hpp"
namespace hami {

// constexpr auto DEFAULT_INIT_CONFIG =
//     "List[InstancesRegister[BackgroundThread[BackendProxy]], "
//     "Register[Aspect[Batching, "
//     "InstanceDispatcher]]]";
constexpr auto DEFAULT_NODE_CONFIG =
    u8"IoC[SharedInstancesState,InstanceDispatcher,Batching;DI[Batching,"
    "InstanceDispatcher]]";
// constexpr auto DEFAULT_NODE_CONFIG = "IoC[Pass;Identity]";
constexpr auto DEFAULT_INSTANCES_CONFIG = u8"BackgroundThread[Reflect[backend]]";
static const auto DEFAULT_INIT_CONFIG = std::string("List[InstancesRegister[") +
                                 DEFAULT_INSTANCES_CONFIG + "], " +
                                 "Register[" + DEFAULT_NODE_CONFIG + "]]";

class Interpreter : public Proxy {
   public:
    void init(const std::unordered_map<std::string, std::string>& config,
              const dict& dict_config) override final;

   private:
    std::vector<std::unique_ptr<Backend>> inited_dependencies_;
};
}  // namespace hami