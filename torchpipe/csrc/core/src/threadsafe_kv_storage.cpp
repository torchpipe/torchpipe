#include <optional>

#include "threadsafe_kv_storage.hpp"
#include "any2object.hpp"
#include "object2any.hpp"

#include "threadsafe_queue.hpp"
#include "base_logging.hpp"

namespace ipipe {

// 获取单例实例
ThreadSafeKVStorage& ThreadSafeKVStorage::getInstance() {
  static ThreadSafeKVStorage instance;
  return instance;
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
    std::unique_lock<std::shared_mutex> lock(mutex_);
    cv_.wait(lock, [this, path, key] {
      auto it = this->disk_.find(path);
      return it != disk_.end() && it->second->has(key);
    });
  }

  return get(path, key);
}

// bool ThreadSafeKVStorage::has(const std::string& path) {
//   std::shared_lock<std::shared_mutex> lock(mutex_);
//   return disk_.find(path) != disk_.end();
// }

ThreadSafeDict& ThreadSafeKVStorage::get_or_insert(const std::string& path) {
  // SPDLOG_DEBUG("ThreadSafeKVStorage: {} {} {}", path, (void*)(&disk_), disk_.size());
  {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    auto it = disk_.find(path);
    if (it != disk_.end()) {
      // SPDLOG_DEBUG("ThreadSafeKVStorage get: {} {} {}", path, disk_.size(),
      // (void*)&(*it->second));
      return *it->second;
    }
  }
  std::unique_lock<std::shared_mutex> lock(mutex_);

  auto [it, inserted] = disk_.emplace(path, std::make_shared<ThreadSafeDict>());
  // SPDLOG_DEBUG("ThreadSafeKVStorage insert: {} {} {}", path, disk_.size(), it->first);
  return *it->second;
}
// 写入数据
void ThreadSafeKVStorage::set_and_notify(const std::string& path, const std::string& key,
                                         ipipe::any value) {
  std::shared_ptr<ThreadSafeDict> data;
  {
    // Use a shared lock for reading
    std::shared_lock<std::shared_mutex> lock(mutex_);
    auto iter = disk_.find(path);
    if (iter != disk_.end()) {
      data = iter->second;
    }
    // else {
    //   throw std::out_of_range(path + " already exists");
    // }
  }

  if (data) {
    // If we found a dict, use it
    data->set(key, value);
  } else {
    // Otherwise, create a new dict and insert it into disk_
    // Use a unique lock for writing
    data = std::make_shared<ThreadSafeDict>();
    data->set(key, value);

    std::unique_lock<std::shared_mutex> lock(mutex_);
    disk_[path] = data;
  }
  cv_.notify_all();
}

void ThreadSafeKVStorage::clear() {
  std::unique_lock<std::shared_mutex> lock(mutex_);
  disk_.clear();
}

void ThreadSafeKVStorage::erase(const std::string& path) {
  std::unique_lock<std::shared_mutex> lock(mutex_);
  auto iter = disk_.find(path);
  if (iter != disk_.end()) {
    disk_.erase(iter);
  } else
    throw std::out_of_range(path);
}

}  // namespace ipipe
// #ifdef PYBIND
// #include <pybind11/pybind11.h>
// #include <pybind11/stl.h>
// #include "tensor_type_caster.hpp"
// #endif
