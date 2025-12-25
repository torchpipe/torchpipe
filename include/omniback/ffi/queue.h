#ifndef OMNIBACK_FFI_QUEUE_H_
#define OMNIBACK_FFI_QUEUE_H_

#include <tvm/ffi/memory.h>
#include <tvm/ffi/any.h>
#include <tvm/ffi/optional.h>
#include <tvm/ffi/error.h>

#include <condition_variable>
#include <deque>
#include <mutex>
#include <chrono>
#include <vector>
#include <utility>
#include <optional>
#include <omniback/core/any.hpp>

namespace omniback {
namespace ffi {

/*!
 * \brief ThreadSafeQueueObj stores queue elements, supports thread-safe push/pop with
 * strong type safety.
 */
class ThreadSafeQueueObj : public tvm::ffi::Object {
 public:
  static constexpr bool _type_mutable = true;

  /*! \brief Put an element to the queue (copy version) */
  template <typename T>
  void put(const T& value) {
    std::lock_guard<std::mutex> lock(mutex_);
    queue_.emplace_back(value);
    cv_not_empty_.notify_all();
  }

  /*! \brief Put an element to the queue (move version) */
  template <typename T>
  void put(T&& value) {
    std::lock_guard<std::mutex> lock(mutex_);
    queue_.emplace_back(std::forward<T>(value));
    cv_not_empty_.notify_all();
  }

  template <typename T>
  void put_wo_notify(const T& value) {
    std::lock_guard<std::mutex> lock(mutex_);
    queue_.emplace_back(value);
  }

  /*! \brief Put multiple elements to the queue (rvalue reference) */
  template <typename T>
  void puts(std::vector<T>&& values) {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& value : values) {
      queue_.emplace_back(std::move(value));
    }
    cv_not_empty_.notify_all();
  }

  /*! \brief Put multiple elements to the queue (const reference) */
  template <typename T>
  void puts(const std::vector<T>& values) {
    std::lock_guard<std::mutex> lock(mutex_);
    for (const auto& value : values) {
      queue_.emplace_back(value);
    }
    cv_not_empty_.notify_all();
  }

  /*! \brief Get an element from the queue (blocking, type-safe) */
  template <typename T>
  T get() {
    std::unique_lock<std::mutex> lock(mutex_);
    cv_not_empty_.wait(lock, [this] { return !queue_.empty(); });
    return unsafe_pop_front<T>();
  }

  template <typename T>
  T unsafe_pop_front(){
    TVM_FFI_ICHECK(!queue_.empty()); 
    omniback::any value = std::move(queue_.front());
    queue_.pop_front();
    return std::move(value).cast<T>();
  }

  template <typename T>
  std::optional<T> pop(int timeout_ms) {
    std::unique_lock<std::mutex> lock(mutex_);

    const auto timeout_dur = std::chrono::milliseconds(timeout_ms);

    bool has_value = cv_not_empty_.wait_for(
        lock, timeout_dur, [this]() { return !queue_.empty(); });

    if (!has_value) {
      return std::nullopt;
    }

    return unsafe_pop_front<T>();
  }

        /*! \brief Get an element with timeout (blocking with timeout, type-safe)
       */
  template <typename T>
  std::optional<T> get(double timeout_seconds) {
    std::unique_lock<std::mutex> lock(mutex_);
    auto timeout = std::chrono::duration<double>(timeout_seconds);
    bool success = cv_not_empty_.wait_for(
        lock, timeout, [this] { return !queue_.empty(); });

    if (!success)
      return std::nullopt;
    return extract_value<T>(std::move(queue_.front()));
  }

  /*! \brief Try to get an element without blocking (type-safe) */
  template <typename T>
  std::optional<T> try_get() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (queue_.empty())
      return std::nullopt;
    return extract_value<T>(std::move(queue_.front()));
  }

  /*! \brief Get front element without removing (type-safe, const access) */
  template <typename T>
  T front() const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (queue_.empty()) {
      TVM_FFI_THROW(tvm::ffi::IndexError)
          << "Cannot access front of empty queue";
    }
    return queue_.front().cast<T>();
  }

  /*! \brief Pop the front element without returning it */
  void pop() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (queue_.empty()) {
      TVM_FFI_THROW(tvm::ffi::IndexError) << "Cannot pop from empty queue";
    }
    queue_.pop_front();
  }

  /*! \brief Get current size of queue */
  size_t size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return queue_.size();
  }

  /*! \brief Check if queue is empty */
  bool empty() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return queue_.empty();
  }

  /*! \brief Clear all elements from queue */
  void clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    queue_.clear();
  }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL(
      "omniback.Queue",
      ThreadSafeQueueObj,
      tvm::ffi::Object);

 private:
  template <typename T>
  T extract_value(omniback::any&& any_val) {
    T value = std::move(any_val).cast<T>();
    queue_.pop_front();
    return value;
  }

  mutable std::mutex mutex_;
  std::condition_variable cv_not_empty_;
  std::deque<omniback::any> queue_;
};

class ThreadSafeQueueRef : public tvm::ffi::ObjectRef {
 public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(
      ThreadSafeQueueRef,
      tvm::ffi::ObjectRef,
      ThreadSafeQueueObj);
};

ThreadSafeQueueObj& default_queue(const std::string& tag = "");

} // namespace ffi
} // namespace omniback

#endif // OMNIBACK_FFI_QUEUE_H_