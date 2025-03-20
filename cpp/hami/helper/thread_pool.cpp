
#include <queue>
#include <mutex>
#include <condition_variable>
#include <optional>
#include <stdexcept>
#include <chrono>
#include <functional>

#include "hami/helper/thread_pool.hpp"

namespace hami::thread_pool {

BS::thread_pool<>& default_thread_pool(const std::string& tag, size_t size) {
  static std::mutex mtx;
  static std::unordered_map<std::string, std::shared_ptr<BS::thread_pool<>>>
      pool_map;

  std::lock_guard<std::mutex> lock(mtx);

  auto iter = pool_map.find(tag);
  if (iter == pool_map.end()) {
    // 使用 try_emplace 避免拷贝（C++17 支持）
    auto [new_iter, _] =
        pool_map.try_emplace(tag, std::make_shared<BS::thread_pool<>>(size));
    return *new_iter->second;
  }

  // 可选：检查现有线程池的 size 是否匹配传入参数
  if (size != 0 && iter->second->get_thread_count() != size) {
    throw std::invalid_argument("Thread pool size mismatch for tag: " + tag);
  }

  return *iter->second;
}

}  // namespace hami::thread_pool