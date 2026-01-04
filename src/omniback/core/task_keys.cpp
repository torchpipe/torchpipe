#include "omniback/core/task_keys.hpp"

namespace omniback {

bool try_replace_inner_key(std::string& key) {
  static const string prefix = "TASK_";
  static const string suffix = "_KEY";
  static const size_t prefix_suffix_len = prefix.size() + suffix.size();

  if (key.size() >= prefix_suffix_len &&
      key.compare(0, prefix.size(), prefix) == 0 &&
      key.compare(key.size() - suffix.size(), suffix.size(), suffix) == 0) {
    const auto iter = TASK_KEY_MAP.find(key);
    if (iter == TASK_KEY_MAP.end()) {
      throw std::runtime_error("Inner key not supported: " + key);
    }
    key = iter->second;
    return true;
  }
  return false;
}
} // namespace omniback