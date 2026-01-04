
#include <chrono>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <optional>
#include <queue>
#include <stdexcept>

#include "omniback/helper/thread_pool.hpp"

namespace omniback::thread_pool {

BS::thread_pool<>& default_thread_pool(const std::string& tag, size_t size) {
  static std::mutex mtx;
  static std::unordered_map<std::string, std::shared_ptr<BS::thread_pool<>>>
      pool_map;

  std::lock_guard<std::mutex> lock(mtx);

  auto iter = pool_map.find(tag);
  if (iter == pool_map.end()) {
    auto [new_iter, _] =
        pool_map.try_emplace(tag, std::make_shared<BS::thread_pool<>>(size));
    return *new_iter->second;
  }

  if (size != 0 && iter->second->get_thread_count() != size) {
    throw std::invalid_argument("Thread pool size mismatch for tag: " + tag);
  }

  return *iter->second;
}

} // namespace omniback::thread_pool