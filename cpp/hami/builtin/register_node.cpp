#include <charconv>
#include "hami/builtin/register_node.hpp"
#include "hami/core/reflect.h"
#include "hami/helper/base_logging.hpp"
#include "hami/core/task_keys.hpp"

namespace hami {
void Register::pre_init(
    const std::unordered_map<std::string, std::string>& config,
    const dict& dict_config) {
    auto iter = config.find("node_name");
    HAMI_ASSERT(iter != config.end(), "node_name not found");
    for (const auto & item: config)
    {
        SPDLOG_INFO("[Register] {}: {}", item.first, item.second);
    }
    set_registered_name("node." + iter->second);
}

HAMI_REGISTER(Backend, Register, "Register,Node");

class InstancesRegister : public Backend {
   public:
    std::shared_ptr<Backend> owned_backend_;
    void init(const std::unordered_map<std::string, std::string>& config,
              const dict& dict_config) override final {
        std::string dependency_name;
        constexpr auto default_name = "InstancesRegister";
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
        if (iter != config.end()) {
            dependency_name = iter->second;
        } else {
            throw std::runtime_error("Dependency configuration  " + *name +
                                     "::dependency not found, skipping "
                                     "dependency injection process");
        }

        iter = config.find("node_name");
        HAMI_ASSERT(iter != config.end(), "`node_name` not found");
        std::string node_name = iter->second;
        iter = config.find("instance_num");
        size_t instance_num{1};
        if (iter == config.end()) {
            SPDLOG_INFO("{}::init, instance_num not found, using default: 1",
                        *name);
        } else {
            auto [ptr, ec] = std::from_chars(
                iter->second.data(), iter->second.data() + iter->second.size(),
                instance_num);
            HAMI_ASSERT(ec == std::errc(),
                        "invalid instance_num: " + iter->second);
        }

        auto sub_dict_config = dict_config ? dict_config : make_dict();
        auto sub_config = config;
        for (size_t i = 0; i < instance_num; ++i) {
            owned_backend_ =
                std::shared_ptr<Backend>(HAMI_CREATE(Backend, dependency_name));
            HAMI_ASSERT(owned_backend_,
                        "`" + iter->second + "` is not a valid backend");
            sub_config[TASK_INDEX_KEY] = std::to_string(i);
            owned_backend_->init(sub_config, sub_dict_config);

            HAMI_INSTANCE_REGISTER(Backend, node_name + "." + std::to_string(i),
                                   owned_backend_);
        }
    }

    void forward(const std::vector<dict>& inputs) override final {
        owned_backend_->forward(inputs);
    }
};

HAMI_REGISTER(Backend, InstancesRegister, "InstancesRegister,Instances");

}  // namespace hami