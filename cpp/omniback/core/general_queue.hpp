#pragma once

#include <chrono>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <numeric>
#include <optional>
#include <queue>
#include <stdexcept>
#include "omniback/core/request_size.hpp"

namespace omniback::queue {

class QueueException : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

class QueueEmptyException : public QueueException {
 public:
  QueueEmptyException() : QueueException("Queue is empty") {}
  using QueueException::QueueException;
};

class QueueFullException : public QueueException {
 public:
  QueueFullException() : QueueException("Queue is full") {}
  using QueueException::QueueException;
};

template <typename T>
class SizedQueue {
 private:
  struct SizedElement {
    T value;
    size_t size;

    SizedElement(const T& val, size_t sz = 1) : value(val), size(sz) {}
  };

  std::queue<SizedElement> queue_;
  size_t totalSize_ = 0;

 public:
  explicit SizedQueue() = default;

  // Push an element with a specified size
  void push(const T& value, size_t size = 1) {
    queue_.push(SizedElement(value, size));
    totalSize_ += size;
  }

  // Pop the front element and reduce the total size
  void pop() {
    if (queue_.empty()) {
      throw QueueEmptyException();
    }
    totalSize_ -= queue_.front().size;
    queue_.pop();
  }

  // Get the front element
  std::pair<const T&, size_t> front() const {
    if (queue_.empty()) {
      throw QueueEmptyException();
    }
    return std::pair<const T&, size_t>(
        queue_.front().value, queue_.front().size);
  }

  // Get the total size of the queue
  size_t size() const {
    return totalSize_;
  }

  // Check if the queue is empty
  bool empty() const {
    return queue_.empty();
  }
};

template <typename T>
class ThreadSafeQueue {
 private:
  std::queue<T> queue_;
  mutable std::mutex mutex_;
  std::condition_variable popped_cond_;
  std::condition_variable pushed_cond_;

 public:
  explicit ThreadSafeQueue() = default;

  // Push an element with a specified size
  template <
      typename U,
      typename = std::enable_if_t<std::is_convertible_v<U, T>>>
  void put(U&& value) {
    {
      std::unique_lock<std::mutex> lock(mutex_);

      queue_.push(std::forward<U>(value));
    }
    pushed_cond_.notify_all();
  }

  template <
      typename Rep,
      typename Period,
      typename U,
      typename = std::enable_if_t<std::is_convertible_v<U, T>>>
  bool try_put(
      U&& value,
      size_t maxSize,
      std::chrono::duration<Rep, Period> timeout) {
    {
      std::unique_lock<std::mutex> lock(mutex_);
      if (!popped_cond_.wait_for(lock, timeout, [this, maxSize] {
            return queue_.size() < maxSize;
          })) {
        // throw QueueFullException("Queue is full after timeout");
        return false;
      }
      queue_.push(std::forward<U>(value));
    }
    pushed_cond_.notify_all();
    return true;
  }

  template <typename Rep, typename Period, template <typename> class Container>
  bool try_put(
      const Container<T>& values,
      size_t maxSize,
      std::chrono::duration<Rep, Period> timeout) {
    {
      if (values.size() > maxSize) {
        throw QueueException("Queue is full: values.size() > maxSize");
      }
      maxSize -= values.size();
      std::unique_lock<std::mutex> lock(mutex_);

      if (!popped_cond_.wait_for(lock, timeout, [this, maxSize] {
            return queue_.size() <= maxSize;
          })) {
        // throw QueueFullException("Queue is full after timeout");
        return false;
      }
      for (const auto& value : values) {
        queue_.push(value);
      }
    }
    pushed_cond_.notify_all();
    return true;
  }

  T get(bool block = true, std::optional<double> timeout = std::nullopt) {
    std::unique_lock<std::mutex> lock(mutex_);

    if (queue_.empty()) {
      if (!block) {
        throw QueueEmptyException();
      } else {
        if (timeout) {
          if (!pushed_cond_.wait_for(
                  lock, std::chrono::duration<double>(*timeout), [this] {
                    return !queue_.empty();
                  })) {
            throw QueueEmptyException("Queue is empty after timeout");
          }
        } else {
          pushed_cond_.wait(lock, [this] { return !queue_.empty(); });
        }
      }
    }
    T tmp;
    std::swap(tmp, queue_.front());
    queue_.pop();
    popped_cond_.notify_one();
    return tmp;
  }

  // Get the front element
  const T& front() const {
    std::unique_lock<std::mutex> lock(mutex_);

    if (queue_.empty()) {
      throw QueueEmptyException();
    }

    return queue_.front();
  }

  // Get the total size of the queue
  size_t size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return queue_.size();
  }

  // Check if the queue is empty
  bool empty() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return queue_.empty();
  }

  // please note that there is no notify when get() called. So call it
  // by yourself.
  template <typename Rep, typename Period>
  bool wait_for_new_data(
      std::function<bool(size_t)> size_condition,
      std::chrono::duration<Rep, Period> timeout) {
    std::unique_lock<std::mutex> lock(mutex_);

    auto predicate = [this, size_condition]() {
      return size_condition(queue_.size());
    };
    return pushed_cond_.wait_for(lock, timeout, predicate);
  }

  template <template <typename> class Container>
  void put_without_notify(const Container<T>& values) {
    std::unique_lock<std::mutex> lock(mutex_);
    for (const auto& value : values) {
      queue_.push(value);
    }
  }

  void notify_one() {
    pushed_cond_.notify_one();
  }
  void notify_all() {
    pushed_cond_.notify_all();
  }

  void popped_notify_one() {
    popped_cond_.notify_one();
  }
  void popped_notify_all() {
    popped_cond_.notify_all();
  }

  // Try to pop an element within a timeout
  template <typename Rep, typename Period>
  std::optional<T> try_get(std::chrono::duration<Rep, Period> timeout) {
    std::unique_lock<std::mutex> lock(mutex_);

    if (!pushed_cond_.wait_for(
            lock, timeout, [this] { return !queue_.empty(); })) {
      return std::nullopt;
    }

    T tmp;
    std::swap(tmp, queue_.front());
    queue_.pop();
    lock.unlock();

    popped_cond_.notify_one();
    return tmp;
  }
};

template <typename T>
class ThreadSafeSizedQueue {
  // queue status
 public:
  enum Status { RUNNING, PAUSED, ERROR, CANCELED, EOS };
  bool is_status(Status status) {
    std::lock_guard<std::mutex> lock(status_mutex_);
    return status_ == status;
  }

  Status status() {
    std::lock_guard<std::mutex> lock(status_mutex_);
    return status_;
  }
  void cancel() {
    std::lock_guard<std::mutex> lock(status_mutex_);
    if (Status::RUNNING == status_ || Status::PAUSED == status_) {
      status_ = Status::CANCELED;
      status_cond_.notify_all();
    }
  }
  void set_status(Status status) {
    {
      std::lock_guard<std::mutex> lock(status_mutex_);
      status_ = status;
    }
    status_cond_.notify_all();
  }
  void set_error(std::exception_ptr excep) {
    {
      std::lock_guard<std::mutex> lock(status_mutex_);
      status_ = Status::ERROR;
      excep_ = excep;
    }

    status_cond_.notify_all();
  }
  void join() {
    {
      std::unique_lock<std::mutex> lock(mutex_);
      // status_cond_.wait(lock, [this]()
      //                   { return Status::RUNNING != status_ &&
      //                   Status::PAUSED != status_; });
      popped_cond_.wait(lock, [this]() { return queue_.empty(); });
    }
    std::lock_guard<std::mutex> lock(status_mutex_);
    if (excep_) {
      status_ = Status::ERROR;
      std::rethrow_exception(excep_);
    }
    status_ = Status::EOS;
  }

  void set_input_callback(std::function<void(void)> f) {
    input_callback_ = f;
  }

 private:
  mutable std::mutex status_mutex_;
  std::condition_variable status_cond_;
  Status status_{Status::RUNNING};
  std::exception_ptr excep_{nullptr};

  std::function<void(void)> input_callback_;

  // queue data
 private:
  struct SizedElement {
    T value;
    size_t size;

    SizedElement(const T& val, size_t sz = 1) : value(val), size(sz) {}
  };

  std::queue<SizedElement> queue_;
  size_t totalSize_ = 0;
  mutable std::mutex mutex_;
  std::condition_variable popped_cond_;
  std::condition_variable pushed_cond_;
  // std::stomic_bool shutdown_{false};

 public:
  explicit ThreadSafeSizedQueue() = default;

  // bool is_statue(Status status) {
  //     std::lock_guard<std::mutex> lock(mutex_);
  //     return status_ == status;
  // }
  // void shutdown() { shutdown_.store(true); }
  // bool is_shutdown() { return shutdown_.load(); }
  // Push an element with a specified size
  template <
      typename U,
      typename = std::enable_if_t<std::is_convertible_v<U, T>>>
  void put(U&& value) {
    {
      std::unique_lock<std::mutex> lock(mutex_);
      queue_.push(SizedElement(std::forward<U>(value), 1));
      totalSize_ += 1;
      if (input_callback_)
        input_callback_();
    }
    pushed_cond_.notify_all();
  }

  template <
      typename U,
      typename = std::enable_if_t<std::is_convertible_v<U, T>>>
  void put(U&& value, std::function<int(const U&)> f) {
    {
      auto size = f(value);
      std::unique_lock<std::mutex> lock(mutex_);

      queue_.push(SizedElement(std::forward<U>(value), size));
      totalSize_ += size;
      if (input_callback_)
        input_callback_();
    }
    pushed_cond_.notify_all();
  }

  T get(bool block = true, std::optional<double> timeout = std::nullopt) {
    std::unique_lock<std::mutex> lock(mutex_);

    if (queue_.empty()) {
      if (!block) {
        throw QueueEmptyException();
      } else {
        if (timeout) {
          if (!pushed_cond_.wait_for(
                  lock, std::chrono::duration<double>(*timeout), [this] {
                    return !queue_.empty();
                  })) {
            throw QueueEmptyException("Queue is empty after timeout");
          }
        } else {
          pushed_cond_.wait(lock, [this] { return !queue_.empty(); });
        }
      }
    }

    auto item = std::move(queue_.front().value);
    totalSize_ -= queue_.front().size;
    queue_.pop();
    popped_cond_.notify_one();
    return item;
  }

  // Get the front element
  std::pair<const T&, size_t> front() const {
    std::unique_lock<std::mutex> lock(mutex_);

    if (queue_.empty()) {
      throw QueueEmptyException();
    }

    return std::pair<const T&, size_t>(
        queue_.front().value, queue_.front().size);
  }

  size_t front_size() const {
    std::unique_lock<std::mutex> lock(mutex_);

    if (queue_.empty()) {
      throw QueueEmptyException();
    }

    return queue_.front().size;
  }

  // Get the total size of the queue
  size_t size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return totalSize_;
  }

  // Check if the queue is empty
  bool empty() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return queue_.empty();
  }

  bool wait_pop(T& item, double timeout) {
    std::unique_lock<std::mutex> lock(mutex_);
    if (!pushed_cond_.wait_for(
            lock, std::chrono::duration<double>(timeout), [this] { // 移除 *
              return !queue_.empty();
            })) {
      return false;
    }

    item = std::move(queue_.front().value); // 移除前面的 auto
    totalSize_ -= queue_.front().size;
    queue_.pop();
    popped_cond_.notify_one();
    return true;
  }

  // bool wait_pop(&T, int timeout_ms){
  //   std::unique_lock<std::mutex> lock(mutex_);
  //   if (pushed_cond_.wait_for(
  //           lock, timeout_ms, [this]() { return totalSize_ > 0; })) {
  //   }
  // }

  // please note that there is no notify when get() called. So call it
  // by yourself.
  template <typename Rep, typename Period>
  bool wait_for_new_data(
      std::function<bool(size_t)> size_condition,
      std::chrono::duration<Rep, Period> timeout) {
    std::unique_lock<std::mutex> lock(mutex_);

    auto predicate = [this, size_condition]() {
      return size_condition(totalSize_);
    };
    return pushed_cond_.wait_for(lock, timeout, predicate);
  }

  template <typename Rep, typename Period>
  bool wait_for(std::chrono::duration<Rep, Period> timeout) {
    std::unique_lock<std::mutex> lock(mutex_);

    return pushed_cond_.wait_for(
        lock, timeout, [this]() { return totalSize_ > 0; });
  }

  template <typename Rep, typename Period>
  bool wait_until_at_most(
      size_t max_size,
      std::chrono::duration<Rep, Period> timeout) {
    std::unique_lock<std::mutex> lock(mutex_);
    return popped_cond_.wait_for(
        lock, timeout, [this, max_size] { return totalSize_ <= max_size; });
  }

  template <typename Rep, typename Period>
  bool wait_until_at_least(
      size_t min_size,
      std::chrono::duration<Rep, Period> timeout) {
    std::unique_lock<std::mutex> lock(mutex_);
    return pushed_cond_.wait_for(
        lock, timeout, [this, min_size] { return totalSize_ >= min_size; });
  }

  template <template <typename> class Container>
  void put_without_notify(
      const Container<T>& values,
      size_t size_per_item = 1) {
    std::unique_lock<std::mutex> lock(mutex_);
    for (const auto& value : values) {
      queue_.push(SizedElement(value, size_per_item));
      totalSize_ += size_per_item;
    }
  }

  void put_without_notify(const T& value) {
    std::unique_lock<std::mutex> lock(mutex_);

    queue_.push(SizedElement(value, 1));
    totalSize_ += 1;
  }

  template <
      typename U,
      template <typename, typename...> class Container,
      typename... Args>
  void puts(const Container<U, Args...>& values) {
    {
      std::unique_lock<std::mutex> lock(mutex_);
      for (const auto& value : values) {
        const auto size_per_item = get_request_size(value);
        queue_.push(SizedElement(value, size_per_item));
        totalSize_ += size_per_item;
      }
      if (input_callback_)
        input_callback_();
    }

    pushed_cond_.notify_all();
  }

  void notify_one() {
    pushed_cond_.notify_one();
  }
  void notify_all() {
    pushed_cond_.notify_all();
  }

  void popped_notify_one() {
    popped_cond_.notify_one();
  }
  void popped_notify_all() {
    popped_cond_.notify_all();
  }

  // Try to pop an element within a timeout
  template <typename Rep, typename Period>
  std::pair<std::optional<T>, size_t> try_get(
      std::chrono::duration<Rep, Period> timeout) {
    {
      std::unique_lock<std::mutex> lock(mutex_);

      if (queue_.empty()) {
        if (!pushed_cond_.wait_for(
                lock, timeout, [this] { return !queue_.empty(); })) {
          return std::pair<std::optional<T>, size_t>(std::nullopt, 0);
        }
      }

      auto item = std::pair<std::optional<T>, size_t>(
          std::move(queue_.front().value), queue_.front().size);
      totalSize_ -= item.second;
      queue_.pop();
      popped_cond_.notify_one();
      return item;
    }
  }

  template <
      typename Rep,
      typename Period,
      typename U,
      template <typename, typename...> class Container,
      typename... Args,
      typename = std::enable_if_t<std::is_convertible_v<U, T>>>
  bool try_puts(
      const Container<U, Args...>& values,
      size_t maxSize,
      std::chrono::duration<Rep, Period> timeout) {
    {
      std::vector<size_t> size(values.size());

      for (size_t i = 0; i < values.size(); ++i) {
        size[i] = get_request_size(values[i]);
      }
      size_t total = std::accumulate(size.begin(), size.end(), 0);

      std::unique_lock<std::mutex> lock(mutex_);

      if (!popped_cond_.wait_for(lock, timeout, [this, total, maxSize] {
            return totalSize_ + total <= maxSize;
          })) {
        return false;
      }
      for (size_t i = 0; i < values.size(); ++i) {
        queue_.push(SizedElement(values[i], size[i]));
      }
      totalSize_ += total;
    }
    pushed_cond_.notify_all();
    return true;
  }

  template <typename Rep, typename Period>
  bool try_put(
      const T& value,
      size_t maxSize,
      std::chrono::duration<Rep, Period> timeout) {
    {
      const size_t size = get_request_size(value);
      std::unique_lock<std::mutex> lock(mutex_);

      if (!popped_cond_.wait_for(lock, timeout, [this, size, maxSize] {
            return totalSize_ + size <= maxSize;
          })) {
        return false;
      }
      queue_.push(SizedElement(value, size));
      totalSize_ += size;
    }
    pushed_cond_.notify_all();
    return true;
  }

  //    public:
  //     void set_on_query(std::function<void(ThreadSafeSizedQueue*)>
  //     on_query) {
  //         on_query_ = on_query;
  //     }

  //    private:
  //     std::function<void(ThreadSafeSizedQueue*)> on_query_;
  // ~ThreadSafeSizedQueue() { shutdown(); }
};

} // namespace omniback::queue