#pragma once

#include <string>
#include <optional>
#include <unordered_map>
#include <memory>
#include <shared_mutex>
#include <mutex>
#include <unordered_map>
#include <shared_mutex>

#include "any.hpp"
namespace ipipe {

class ThreadSafeDict {
 private:
  std::unordered_map<std::string, ipipe::any> map_;
  std::shared_mutex mutex_;

 public:
  const std::optional<ipipe::any> get(const std::string& key) {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    if (map_.find(key) == map_.end()) {
      return std::nullopt;
    }
    return map_[key];
  }

  void set(const std::string& key, const ipipe::any& value) {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    map_[key] = value;
  }

  void erase(const std::string& key) {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    map_.erase(key);
  }
};

class ThreadSafeKVStorage {
 public:
  // 获取单例实例
  static ThreadSafeKVStorage& getInstance();

  // 删除拷贝构造函数和赋值操作符，防止复制单例对象
  ThreadSafeKVStorage(const ThreadSafeKVStorage&) = delete;
  ThreadSafeKVStorage& operator=(const ThreadSafeKVStorage&) = delete;

  ThreadSafeDict& get(const std::string& path) {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    auto it = disk_.find(path);
    if (it != disk_.end()) {
      return *it->second;
    }
    throw std::out_of_range("ThreadSafeKVStorage: Key not found: " + path);
  }

  ThreadSafeDict& get_or_insert(const std::string& path) {
    {
      std::shared_lock<std::shared_mutex> lock(mutex_);
      auto it = disk_.find(path);
      if (it != disk_.end()) {
        return *it->second;
      }
    }
    std::unique_lock<std::shared_mutex> lock(mutex_);
    auto [it, inserted] = disk_.emplace(path, std::make_shared<ThreadSafeDict>());
    return *it->second;
  }

  // 读取数据
  std::optional<ipipe::any> get(const std::string& path, const std::string& key);

  // 写入数据
  void set(const std::string& path, const std::string& key, ipipe::any data);

  template <typename T>
  void set(const std::string& path, const std::string& key, T data) {
    set(path, key, ipipe::any(data));
  }

  // 清空数据
  void clear();

  void erase(const std::string& path);

 private:
  ThreadSafeKVStorage() = default;

  // any has a pybind11 binding： Any
  std::unordered_map<std::string, std::shared_ptr<ThreadSafeDict>> disk_;
  std::shared_mutex mutex_;
};

}  // namespace ipipe
