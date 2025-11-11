#pragma once

#include "omniback/core/dict.hpp"
// constexpr auto TASK_REQUEST_KEY = "request";
constexpr auto TASK_REQUEST_SIZE_KEY = "request_size";

namespace omniback {
template <typename Container = std::vector<dict>>
static inline int get_request_size(const Container& in) {
  int total_len = 0;
  for (const auto& item : in) {
    total_len += get_request_size(item);
  }
  return total_len;
}

template <>
inline int get_request_size<dict>(const dict& in) {
  const auto iter = in->find(TASK_REQUEST_SIZE_KEY);
  if (iter != in->end()) {
    if (iter->second.type() == typeid(int))
      return any_cast<int>(iter->second);
    // else if (iter->second.type() == typeid(size_t))
    // return any_cast<size_t>(iter->second);
    else {
      return std::stoi(any_cast<string>(iter->second));
    }
  } else {
    return 1;
  }
}
} // namespace omniback