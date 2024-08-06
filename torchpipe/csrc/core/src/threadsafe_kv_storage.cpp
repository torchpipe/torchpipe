#include <optional>

#include "threadsafe_kv_storage.hpp"
#include "any2object.hpp"
#include "object2any.hpp"

#include "threadsafe_queue.hpp"

namespace ipipe {

// 获取单例实例
ThreadSafeKVStorage& ThreadSafeKVStorage::getInstance() {
  static ThreadSafeKVStorage instance;
  return instance;
}

std::optional<ipipe::any> ThreadSafeKVStorage::get(const std::string& path,
                                                   const std::string& key) {
  std::shared_ptr<ThreadSafeDict> dict;
  {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    auto it = disk_.find(path);
    if (it != disk_.end()) {
      dict = it->second;
    } else {
      return std::nullopt;
    }
  }

  return dict->get(key);
}

// bool ThreadSafeKVStorage::has(const std::string& path) {
//   std::shared_lock<std::shared_mutex> lock(mutex_);
//   return disk_.find(path) != disk_.end();
// }

// 写入数据
void ThreadSafeKVStorage::set(const std::string& path, const std::string& key, ipipe::any value) {
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
