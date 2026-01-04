#pragma once
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <unordered_map>
#include <vector>

namespace omniback {

template <typename Key, typename Value>
class threadsafe_unordered_map {
 private:
  std::unordered_map<Key, Value> map_;
  mutable std::shared_mutex mutex_;

 public:
  // 线程安全的 operator[] 实现
  Value& operator[](const Key& key) {
    std::unique_lock lock(mutex_);
    return map_[key]; // 如果不存在会自动插入默认构造的值
  }

  // const版本不提供 operator[]，因为可能意外创建元素
  // 替代方案：提供 at() 方法
  const Value& at(const Key& key) const {
    std::shared_lock lock(mutex_);
    return map_.at(key); // 如果不存在会抛出 std::out_of_range
  }

  // 线程安全的访问或插入
  Value& operator[](Key&& key) {
    std::unique_lock lock(mutex_);
    return map_[std::move(key)];
  }

  // 插入或更新元素
  void insert_or_assign(const Key& key, const Value& value) {
    std::unique_lock lock(mutex_);
    map_[key] = value;
  }

  std::vector<Key> keys() const {
    std::shared_lock lock(mutex_);
    std::vector<Key> result;
    result.reserve(map_.size());
    for (const auto& pair : map_) {
      result.push_back(pair.first);
    }
    return result;
  }

  // 获取元素
  Value get(const Key& key) const {
    std::shared_lock lock(mutex_);
    return map_.at(key);
  }

  // 查找元素
  auto find(const Key& key) const {
    std::shared_lock lock(mutex_);
    return map_.find(key);
  }

  // 线程安全的迭代器访问（返回整个容器的拷贝）
  auto snapshot() const {
    std::shared_lock lock(mutex_);
    return map_; // 返回整个map的拷贝
  }

  // 以下迭代器方法需要特别注意线程安全
  // 注意：这些迭代器只在锁的生命周期内有效

  // 获取起始迭代器（const版本）
  auto begin() const {
    std::shared_lock lock(mutex_);
    return map_.begin();
  }

  // 获取结束迭代器（const版本）
  auto end() const {
    std::shared_lock lock(mutex_);
    return map_.end();
  }

  // 获取起始迭代器（非const版本）
  auto begin() {
    std::unique_lock lock(mutex_);
    return map_.begin();
  }

  // 获取结束迭代器（非const版本）
  auto end() {
    std::unique_lock lock(mutex_);
    return map_.end();
  }

  // cbegin/cend
  auto cbegin() const {
    std::shared_lock lock(mutex_);
    return map_.cbegin();
  }

  auto cend() const {
    std::shared_lock lock(mutex_);
    return map_.cend();
  }

  // 其他原有方法保持不变...
  bool contains(const Key& key) const {
    std::shared_lock lock(mutex_);
    return map_.find(key) != map_.end();
  }

  bool erase(const Key& key) {
    std::unique_lock lock(mutex_);
    return map_.erase(key) > 0;
  }

  size_t size() const {
    std::shared_lock lock(mutex_);
    return map_.size();
  }

  void clear() {
    std::unique_lock lock(mutex_);
    map_.clear();
  }

  bool empty() const {
    std::shared_lock lock(mutex_);
    return map_.empty();
  }

  // 原子操作
  template <typename... Args>
  bool emplace(Args&&... args) {
    std::unique_lock lock(mutex_);
    return map_.emplace(std::forward<Args>(args)...).second;
  }
};
} // namespace omniback