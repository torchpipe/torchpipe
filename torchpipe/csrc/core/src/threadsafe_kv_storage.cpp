#include <optional>

#include "threadsafe_kv_storage.hpp"
#include "any2object.hpp"
#include "object2any.hpp"

#include "threadsafe_queue.hpp"
#include "base_logging.hpp"

namespace ipipe {

std::mutex ThreadSafeKVStorage::instance_mutex_;  // 定义静态互斥锁
std::unordered_map<ThreadSafeKVStorage::POOL, std::unique_ptr<ThreadSafeKVStorage>>
    ThreadSafeKVStorage::instances_;  // 定义静态实例存储

std::unique_ptr<ThreadSafeKVStorage> ThreadSafeKVStorage::createInstance() {
  return std::unique_ptr<ThreadSafeKVStorage>(new ThreadSafeKVStorage());
  // return std::make_unique<ThreadSafeKVStorage>();
}

ThreadSafeKVStorage& ThreadSafeKVStorage::getInstance(POOL pool) {
  std::lock_guard<std::mutex> lock(instance_mutex_);  // 保护实例访问的互斥锁
  if (instances_.find(pool) == instances_.end()) {
    instances_[pool] = createInstance();
  }
  return *instances_[pool];
}

std::optional<ipipe::any> ThreadSafeKVStorage::get(const std::string& path,
                                                   const std::string& key) {
  std::shared_ptr<ThreadSafeDict> dict_data;
  {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    auto it = disk_.find(path);
    if (it != disk_.end()) {
      dict_data = it->second;
    } else {
      return std::nullopt;
    }
  }

  return dict_data->get(key);
}

ipipe::any ThreadSafeKVStorage::wait(const std::string& path, const std::string& key) {
  {
    throw std::runtime_error("ThreadSafeKVStorage::wait not implemented");
    return ipipe::any();
  }
}

ThreadSafeDict& ThreadSafeKVStorage::get_or_insert(const std::string& path) {
  std::unique_lock<std::shared_mutex> lock(mutex_);
  auto it = disk_.find(path);
  if (it != disk_.end()) {
    return *it->second;
  }

  // std::unique_lock<std::shared_mutex> lock(mutex_);

  auto [it_emplace, inserted] = disk_.emplace(path, std::make_shared<ThreadSafeDict>());
  cv_.notify_all();
  return *it_emplace->second;
}

// 写入数据
void ThreadSafeKVStorage::set(const std::string& path, const std::string& key, ipipe::any value) {
  ThreadSafeDict& data = get_or_insert(path);
  data.set(key, value);
}

void ThreadSafeKVStorage::clear() {
  std::unique_lock<std::shared_mutex> lock(mutex_);
  disk_.clear();
}

void ThreadSafeKVStorage::remove(const std::string& path) {
  std::unique_lock<std::shared_mutex> lock(mutex_);
  auto iter = disk_.find(path);
  if (iter != disk_.end()) {
    disk_.erase(iter);
    cv_.notify_all();
  } else {
    SPDLOG_WARN("ThreadSafeKVStorage: key not found: {}", path);
  }

  if (remove_callback_) {
    remove_callback_(path);
  }
  // throw std::out_of_range(path);
}

}  // namespace ipipe
// #ifdef PYBIND
// #include <pybind11/pybind11.h>
// #include <pybind11/stl.h>
// #include "tensor_type_caster.hpp"
// #endif
