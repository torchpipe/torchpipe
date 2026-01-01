#include "omniback/core/reflect.h"
#include "omniback/core/backend.hpp"
#include "omniback/core/event.hpp"
#include "omniback/core/queue.hpp"
#include "omniback/helper/base_logging.hpp"
#include "omniback/helper/string.hpp"

namespace omniback {

bool omniback_load() {
  const static auto tmp = []() { return true; }();
  return tmp;
}

void printlog_and_throw(std::string name) {
  SPDLOG_INFO(name);
  throw std::runtime_error(name);
}
void printlog(std::string name) {
  SPDLOG_INFO(name);
}

void print_check_distance(
    std::string strtem,
    const std::vector<std::string>& targets) {
  std::string re;
  size_t min_distance = std::numeric_limits<uint32_t>::max();
  std::string all_items;
  for (auto& target : targets) {
    all_items += target + ";";
    size_t local_distance = str::edit_distance(strtem, target);
    if (local_distance < min_distance) {
      min_distance = local_distance;
      re = target;
    }
    if (min_distance < 4 && min_distance <= strtem.length() / 2)
      SPDLOG_WARN("{} not found. Did you mean {}?", strtem, re);
    SPDLOG_DEBUG("all = {}", all_items);
  }
}

OMNI_EXPORT std::vector<std::string> strict_str_split(
    std::string strtem,
    char a) {
  str::remove_space_and_ctrl(strtem);

  if (strtem.empty())
    return {};

  std::vector<std::string> strvec;

  std::string::size_type pos1, pos2;
  pos2 = strtem.find(a);
  pos1 = 0;
  while (std::string::npos != pos2) {
    strvec.push_back(strtem.substr(pos1, pos2 - pos1));

    pos1 = pos2 + 1;
    pos2 = strtem.find(a, pos1);
  }
  strvec.push_back(strtem.substr(pos1));
  for (auto iter_vec = strvec.begin(); iter_vec != strvec.end();) {
    // OMNI_ASSERT(!iter_vec->empty(), "src=" + strtem + ",char=" + a);
    if (iter_vec->empty())
      iter_vec = strvec.erase(iter_vec);
    else
      ++iter_vec;
  }
  OMNI_ASSERT(!strvec.empty());
  return strvec;
}

std::vector<std::array<std::string, 2>> multi_str_split(
    std::string strtem,
    char inner,
    char outer) {
  auto re = std::vector<std::array<std::string, 2>>();
  auto data = strict_str_split(strtem, outer);
  for (auto& item : data) {
    auto tmp = strict_str_split(item, inner);
    OMNI_ASSERT(tmp.size() == 2);
    re.push_back({tmp[0], tmp[1]});
  }
  return re;
}

template <typename RegistryType>
ClassRegistryBase<RegistryType>& ClassRegistryInstance() {
  static ClassRegistryBase<RegistryType> class_register;
  // SPDLOG_INFO(" class registry base: addr = {}", (long
  // long)&(class_register));
  return class_register;
}

// 显式实例化所需类型
template ClassRegistryBase<Backend>& ClassRegistryInstance<Backend>();
template ClassRegistryBase<Queue>& ClassRegistryInstance<Queue>();
template ClassRegistryBase<Event>& ClassRegistryInstance<Event>();

} // namespace omniback
