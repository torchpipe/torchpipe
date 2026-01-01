#include "omniback/builtin/register_node.hpp"

#include <charconv>

#include "omniback/core/reflect.h"
#include "omniback/core/task_keys.hpp"
#include "omniback/helper/base_logging.hpp"

namespace omniback {
void Register::impl_init(
    const std::unordered_map<std::string, std::string>& params,
    const dict& kwargs) {
  parser_v2::ArgsKwargs args_kwargs =
      parser_v2::get_args_kwargs(this, "Register", params);
  std::string node_name = str::get(args_kwargs.second, "node_name");
  std::string backend_name =
      parser_v2::get_dependency_name(this, args_kwargs.second, "Register");
  auto* dep = OMNI_CREATE(Backend, backend_name, "node." + node_name);
  OMNI_ASSERT(dep);
  owned_backend_ = std::unique_ptr<Backend>(dep);
  dep->init(params, kwargs);
  inject_dependency(dep);

  for (const auto& item : args_kwargs.second) {
    SPDLOG_DEBUG(
        "{}: {} [{}, {}]", item.first, item.second, dep->min(), dep->max());
  }
  // set_registered_name("node." + node_name);
}

OMNI_REGISTER(Backend, Register, "Register,Node");

class InstancesRegister : public Backend {
 private:
  // std::shared_ptr<Backend> owned_backend_;

 public:
  void impl_init(
      const std::unordered_map<std::string, std::string>& config,
      const dict& kwargs) override final {
    std::string dependency_name;
    constexpr auto default_name = "InstancesRegister";
    auto name = OMNI_OBJECT_NAME(Backend, this);
    if (name == std::nullopt) {
      name = default_name;
      SPDLOG_WARN(
          "{}::init, it seems this instance was not created via "
          "reflection, using default name {}. "
          "Please configure its dependency via the parameter "
          "{}::dependency",
          *name,
          *name,
          *name);
    }
    auto iter = config.find(*name + "::dependency");
    if (iter != config.end()) {
      dependency_name = iter->second;
    } else {
      throw std::runtime_error(
          "Dependency configuration  " + *name +
          "::dependency not found, skipping "
          "dependency injection process");
    }

    iter = config.find("node_name");
    OMNI_ASSERT(iter != config.end(), "`node_name` not found");
    std::string node_name = iter->second;
    iter = config.find("instance_num");
    size_t instance_num{1};
    if (iter == config.end()) {
      SPDLOG_INFO("{}::init, instance_num not found, using default: 1", *name);
    } else {
      auto [ptr, ec] = std::from_chars(
          iter->second.data(),
          iter->second.data() + iter->second.size(),
          instance_num);
      OMNI_ASSERT(ec == std::errc(), "invalid instance_num: " + iter->second);
    }

    // auto sub_kwargs = kwargs ? kwargs : make_dict();
    auto sub_kwargs = copy_dict(kwargs);
    auto sub_config = config;
    for (size_t i = 0; i < instance_num; ++i) {
      auto owned_backend =
          std::shared_ptr<Backend>(OMNI_CREATE(Backend, dependency_name));
      OMNI_ASSERT(
          owned_backend, "`" + iter->second + "` is not a valid backend");
      sub_config[TASK_INDEX_KEY] = std::to_string(i);
      owned_backend->init(sub_config, sub_kwargs);

      OMNI_INSTANCE_REGISTER(
          Backend, node_name + "." + std::to_string(i), owned_backend);
    }
    // owned_backend_ = nullptr;
  }

  void impl_forward(const std::vector<dict>& inputs) override final {
    OMNI_ASSERT(false, "Not Impl.");
    // owned_backend_->forward(inputs);
  }
};

OMNI_REGISTER(Backend, InstancesRegister, "InstancesRegister,Instances");

} // namespace omniback