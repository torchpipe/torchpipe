#include "omniback/builtin/generate_backend.hpp"
#include <optional>
#include "omniback/core/reflect.h"

namespace omniback {

std::string GenerateBackend::get_latest_backend(
    const std::string& setting) const {
  // A=>A;A[B,C]=>C;A[B,C[D]] => D;

  auto brackets = str::flatten_brackets(setting);
  OMNI_ASSERT(brackets.size() > 0);
  if (brackets.size() == 1 && brackets[0] == setting) {
    return setting;
  }
  auto item = brackets.back();
  auto delim = str::findValidSeparators(item, ';');
  if (!delim.empty())
    (item = item.substr(delim.back() + 1));
  auto items = str::items_split(item, ',');
  OMNI_ASSERT(items.size() > 0);
  return get_latest_backend(items.back());
}

void GenerateBackend::replace_dependency_config(
    std::unordered_map<std::string, std::string>& config,
    std::string dst_key) {
  do {
    auto name = OMNI_OBJECT_NAME(Backend, this);
    if (name == std::nullopt) {
      break;
    }

    auto iter = config.find(*name + "::dependency");
    if (iter == config.end())
      break;
    auto src_key = *name + "::dependency";
    dst_key = dst_key + "::dependency";
    auto src_value = iter->second;
    config.erase(iter);
    config.insert_or_assign(dst_key, src_value);

  } while (false);
}

void GenerateBackend::impl_init(
    const std::unordered_map<std::string, std::string>& const_config,
    const dict& kwargs) {
  auto orders = order();
  std::unordered_map<std::string, std::string> order_config =
      parse_order_config(orders.second);

  auto config = const_config;
  // handle *::dependency: Z=A=>A;Z=A[B,C]=>C;Z=A[B,C[D]] => D;
  replace_dependency_config(config, get_latest_backend(orders.first));
  for (const auto& pair : config) {
    order_config.insert_or_assign(pair.first, pair.second);
  }
  proxy_backend_ = init_backend(orders.first, order_config, kwargs);
  OMNI_ASSERT(proxy_backend_, "GenerateBackend init failed");
}

std::unordered_map<std::string, std::string> GenerateBackend::
    parse_order_config(const std::string& setting) {
  return str::map_split(setting, '=', ',');
}

} // namespace omniback