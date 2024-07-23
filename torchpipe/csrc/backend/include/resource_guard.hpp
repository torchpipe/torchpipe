#pragma once
#include <mutex>
#include <unordered_map>
#include <cassert>

namespace ipipe {
template <typename T>
class ForwardGuard {
 public:
  ForwardGuard(const T &data, void *stream) : stream_(stream) {
    std::lock_guard<std::mutex> lock(forward_data_mutex_);
    assert(forward_data_.find(stream) == forward_data_.end());
    forward_data_.emplace(stream, &data);
  }

  static const T *query_input(void *stream) {
    std::lock_guard<std::mutex> lock(forward_data_mutex_);
    auto it = forward_data_.find(stream);
    if (it == forward_data_.end()) {
      return nullptr;
    }
    return it->second;
  }

  ~ForwardGuard() {
    std::lock_guard<std::mutex> lock(forward_data_mutex_);
    forward_data_.erase(stream_);
  }

 private:
  void *stream_ = nullptr;
  static std::unordered_map<void *, const T *> forward_data_;
  static std::mutex forward_data_mutex_;
};

template <typename T>
std::unordered_map<void *, const T *> ForwardGuard<T>::forward_data_;
template <typename T>
std::mutex ForwardGuard<T>::forward_data_mutex_;

template <typename T>
class ResourceGuard {
 public:
  void add(const T &data, void *stream) {
    std::lock_guard<std::mutex> lock(data_mutex_);
    assert(data_.find(stream) == data_.end());
    data_.emplace(stream, &data);
  }

  const T *query(void *stream) {
    std::lock_guard<std::mutex> lock(data_mutex_);
    auto it = data_.find(stream);
    if (it == data_.end()) {
      return nullptr;
    }
    return it->second;
  }

  void clear() {
    std::lock_guard<std::mutex> lock(data_mutex_);
    data_.clear();
  }

 private:
  std::unordered_map<void *, const T *> data_;
  std::mutex data_mutex_;
};
}  // namespace ipipe