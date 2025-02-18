
#include "hami/builtin/basic_backends.hpp"
#include "hami/core/reflect.h"
#include "hami/helper/macro.h"
#include "hami/helper/base_logging.hpp"
#include "hami/helper/string.hpp"
#include "hami/core/helper.hpp"
#include "hami/builtin/aspect.hpp"
#include "hami/core/task_keys.hpp"

namespace hami {

void Aspect::post_init(const std::unordered_map<std::string, std::string>& config,
                       const dict& dict_config) {
  // set aspect name prefix
  std::string prefix;
  bool has_prefix = false;
  auto iter = config.find("node_name");
  if (iter != config.end()) {
    prefix += iter->second + ".";
  }
  iter = config.find("prefix");
  if (iter != config.end()) {
    prefix = iter->second + ".";
    has_prefix = true;
  } else {
    SPDLOG_DEBUG("{}: prefix not found, will not register the instances", prefix);
  }

  for (size_t i = 0; i < base_dependencies_.size() - 1; ++i) {
    base_dependencies_[i]->inject_dependency(base_dependencies_[i + 1].get());

    if (has_prefix) {
      HAMI_ASSERT(base_config_[i].find("backend") != config.end());
      const auto& backend = base_config_[i]["backend"];
      HAMI_INSTANCE_REGISTER(Backend, prefix + backend, base_dependencies_[i].get());
      SPDLOG_INFO("Aspect::post_init, register backend `{}`.", prefix + backend);
    }
  }

  HAMI_ASSERT(base_dependencies_.size() > 0);
}

void Aspect::forward(const std::vector<dict>& input_output) {
  DictHelper dicts_guard(input_output);
  dicts_guard.keep(TASK_DATA_KEY).erase(TASK_RESULT_KEY);
  base_dependencies_.front()->safe_forward(input_output);
}

std::pair<size_t, size_t> Aspect::update_min_max(const std::vector<Backend*>& depends) {
  // union
  size_t max_value = base_dependencies_.front()->max();
  size_t min_value = base_dependencies_.front()->min();

  return {min_value, max_value};
}

HAMI_REGISTER(Backend, Aspect);

}  // namespace hami
