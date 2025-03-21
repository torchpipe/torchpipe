#include "hami/builtin/interpreter.hpp"

#include "hami/core/parser.hpp"
#include "hami/core/task_keys.hpp"
#include "hami/helper/base_logging.hpp"
#include "hami/helper/macro.h"

namespace hami {
void Interpreter::impl_init(
    const std::unordered_map<std::string, std::string>& config,
    const dict& kwargs) {
  // get the dual config from parameter or kwargs
  auto new_kwargs = kwargs ? kwargs : make_dict();
  auto iter = new_kwargs->find(TASK_CONFIG_KEY);
  if (iter == new_kwargs->end()) {
    (*new_kwargs)[TASK_CONFIG_KEY] = str::mapmap{{TASK_GLOBAL_KEY, config}};
    init(config, new_kwargs);
    return;
  }
  str::mapmap dual_config = any_cast<str::mapmap>(iter->second);

  // parser configuration
  parser::broadcast_global(dual_config);
  auto node_names = parser::set_node_name(dual_config);
  HAMI_FATAL_ASSERT(
      node_names.size() > 0, "Interpreter: no node found in config");

  // per-node setup fron `init`
  for (const auto& item : dual_config) {
    if (item.first == TASK_GLOBAL_KEY)
      continue;
    std::string init_config;
    auto iter_init = item.second.find("init");
    if (iter_init != item.second.end()) {
      init_config = iter_init->second;
    } else {
      SPDLOG_INFO(
          "Interpreter: `init` not found in node config, using default: "
          "{}",
          DEFAULT_INIT_CONFIG);
      init_config = DEFAULT_INIT_CONFIG;
    }
    (*new_kwargs)[TASK_CONFIG_KEY] = dual_config;
    auto tmp = init_backend(init_config, item.second, new_kwargs);
    SPDLOG_INFO(
        "Interpreter: node({})[{} {}]", item.first, tmp->min(), tmp->max());
    inited_dependencies_.emplace_back(std::move(tmp));
  }

  // setup entrypoint
  auto global_config = parser::get_global_config(dual_config);
  auto iter_entry = global_config.find(TASK_ENTRY_KEY);
  std::string entry_name;
  if (iter_entry != global_config.end()) {
    entry_name = iter_entry->second;
  } else {
    entry_name = parser::count(dual_config) == 1
        ? "Forward[node." + *node_names.begin() + "]"
        : "DagDispatcher";
    SPDLOG_INFO(
        "Interpreter: `entrypoint` not found in global config, using "
        "default: {}",
        entry_name);
  }

  owned_backend_ = init_backend(entry_name, global_config, new_kwargs);
  SPDLOG_INFO(
      "Interpreter: entrypoint({})[{} {}]",
      entry_name,
      owned_backend_->min(),
      owned_backend_->max());
  proxy_backend_ = owned_backend_.get();
}

HAMI_REGISTER(Backend, Interpreter);
} // namespace hami