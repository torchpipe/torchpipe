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
#include <type_traits>
#include "omniback/ffi/dict.h"

namespace omniback {
namespace ffi {

/*!
 * \brief ThreadSafeQueueObj stores queue elements, supports thread-safe
 * push/pop with strong type safety and configurable capacity bounds.
 */
class ThreadSafeQueueObj : public tvm::ffi::Object {
 public:
  static constexpr bool _type_mutable = true;

  /*! \brief Default constructor (unbounded queue) */
  ThreadSafeQueueObj() : max_size_(0) {} // 0 means unbounded

  /*! \brief Constructor with capacity bound */
  explicit ThreadSafeQueueObj(size_t max_size) : max_size_(max_size) {}

  /*! \brief Put an element to the queue (copy version) */
  template <typename T>
  void push(const T& value) {
    {
      std::unique_lock<std::mutex> lock(mutex_);
      if (max_size_ > 0) {
        cv_popped_.wait(lock, [this] { return queue_.size() < max_size_; });
      }
      queue_.emplace_back(value);
    }
    cv_pushed_.notify_all();
  }

  void unsafe_lazy_update_size(){
    if (queue_sizes_.size() < queue_.size()){
        real_size_ += queue_.size() - queue_sizes_.size();
        queue_sizes_.resize(queue_.size(),1);
      }
  }

  void notify_one(){
    cv_pushed_.notify_one();
  }

  void popped_notify_one(){
    cv_popped_.notify_one();
  }

  template <typename T>
  void push_with_size(const T& value, size_t real_size) {
    {
      std::unique_lock<std::mutex> lock(mutex_);
      unsafe_lazy_update_size();
      if (max_size_ > 0) {
        cv_popped_.wait(lock, [this, real_size] { return real_size_ + real_size <= max_size_; });
      }
      queue_.emplace_back(value);
      real_size_ += real_size;
      queue_sizes_.push_back(real_size);
    }
    cv_pushed_.notify_all();
  }

  template <typename T>
  bool push_with_max_limit(const T& value,size_t max_size, size_t timeout_ms) {
    TVM_FFI_ICHECK(max_size>0);
    const auto timeout_dur = std::chrono::milliseconds(timeout_ms);
    {
      std::unique_lock<std::mutex> lock(mutex_);

      if (!cv_popped_.wait_for(lock, timeout_dur, [this, max_size] {
            return queue_.size() < max_size;
          }))
        return false;
      queue_.emplace_back(value);
    }
    cv_pushed_.notify_all();
    return true;
  }

  /*! \brief Put an element to the queue (move version) */
  template <typename T>
  void push(T&& value) {
    {
      std::unique_lock<std::mutex> lock(mutex_);
      if (max_size_ > 0) {
        cv_popped_.wait(lock, [this] { return queue_.size() < max_size_; });
      }
      queue_.emplace_back(std::forward<T>(value));
    }
    cv_pushed_.notify_all();
  }

  template <typename T>
  void push_with_size(T&& value, size_t real_size) {
    {
      std::unique_lock<std::mutex> lock(mutex_);
      unsafe_lazy_update_size();
      if (max_size_ > 0) {
        cv_popped_.wait(lock, [this, real_size] { return real_size_ + real_size <= max_size_; });
      }
      queue_.emplace_back(std::forward<T>(value));
      real_size_ += real_size;
      queue_sizes_.push_back(real_size);
    }
    cv_pushed_.notify_all();
  }

  /*! \brief Put without notification (for batch operations) */
  template <typename T>
  void push_wo_notify(T&& value) {
    std::unique_lock<std::mutex> lock(mutex_);
    if (max_size_ > 0) {
      cv_popped_.wait(lock, [this] { return queue_.size() < max_size_; });
    }
    queue_.emplace_back(std::forward<T>(value));
  }

  template <typename Container>
  void pushes(Container&& values) {
    if (values.empty())
      return;

    const size_t num_values = values.size();
    {
      std::unique_lock<std::mutex> lock(mutex_);

      if (max_size_ > 0) {
        cv_popped_.wait(lock, [this, num_values] {
          return queue_.size() + num_values <= max_size_;
        });
      }

      for (auto&& value : values) {
        queue_.emplace_back(std::forward<decltype(value)>(value));
      }
    }

    cv_pushed_.notify_all();
  }

  template <typename Container>
  bool try_pushes(Container&& values, size_t max_size, size_t timeout_ms) {
    TVM_FFI_ICHECK(!values.empty());

    const size_t num_values = values.size();
    {
      const auto timeout_dur = std::chrono::milliseconds(timeout_ms);
      std::unique_lock<std::mutex> lock(mutex_);

      if (max_size > 0) {
        if(!cv_popped_.wait_for(lock, timeout_dur, [this, num_values, max_size] {
          return queue_.size() + num_values <= max_size;
        })) return false;
      }

      for (auto&& value : values) {
        queue_.emplace_back(std::forward<decltype(value)>(value));
      }
    }

    cv_pushed_.notify_all();
    return true;
  }

  template <typename T>
  bool try_push(T&& value, size_t max_size, size_t timeout_ms) {

    {
      const auto timeout_dur = std::chrono::milliseconds(timeout_ms);
      std::unique_lock<std::mutex> lock(mutex_);

      if (max_size > 0) {
        if (!cv_popped_.wait_for(
                lock, timeout_dur, [this, max_size] {
                  return queue_.size() < max_size;
                }))
          return false;
      }

      queue_.emplace_back(std::forward<T>(value));
    }

    cv_pushed_.notify_all();
    return true;
  }

  /*! \brief Get with timeout (type-safe) */
  template <typename T = omniback::any>
  std::optional<T> try_get(size_t timeout_ms) {
    const auto timeout_dur = std::chrono::milliseconds(timeout_ms);

    std::unique_lock<std::mutex> lock(mutex_);
    if (!cv_pushed_.wait_for(
            lock, timeout_dur, [this] { return !queue_.empty(); })) {
      return std::nullopt;
    }

    return unsafe_pop_rm_front<T>();
  }

  bool wait_for(size_t timeout_ms) {
    const auto timeout_dur = std::chrono::milliseconds(timeout_ms);

    std::unique_lock<std::mutex> lock(mutex_);
    return cv_pushed_.wait_for(
            lock, timeout_dur, [this] { return !queue_.empty(); });
  }

  bool wait_until_at_least(size_t queue_size, size_t timeout_ms) {
    const auto timeout_dur = std::chrono::milliseconds(timeout_ms);

    std::unique_lock<std::mutex> lock(mutex_);
    // unsafe_lazy_update_size();
    if (!cv_pushed_.wait_for(lock, timeout_dur, [this, queue_size] {
          return queue_.size() >= queue_size;
        }))
      return false;
    return true;
  }

  template <typename T = omniback::any>
  std::optional<T> try_get(size_t timeout_ms, size_t& out_size) {
    const auto timeout_dur = std::chrono::milliseconds(timeout_ms);

    std::unique_lock<std::mutex> lock(mutex_);
    if (!cv_pushed_.wait_for(
            lock, timeout_dur, [this] { return !queue_.empty(); })) {
      return std::nullopt;
    }

    return unsafe_pop_rm_front<T>(out_size);
  }

  template <typename T = omniback::any>
  T wait_get() {
    std::unique_lock<std::mutex> lock(mutex_);
    cv_pushed_.wait(
            lock, [this] { return !queue_.empty(); });

    return unsafe_pop_rm_front<T>();
  }

  template <typename T = omniback::any>
      T get() {
    std::unique_lock<std::mutex> lock(mutex_);

    return unsafe_pop_rm_front<T>();
  }

  template <typename T = omniback::any>
  T get(size_t& out_size) {
    std::unique_lock<std::mutex> lock(mutex_);

    return unsafe_pop_rm_front<T>(out_size);
  }

  template <typename T = omniback::any>
  T front() const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (queue_.empty()) {
      TVM_FFI_THROW(tvm::ffi::IndexError)
          << "Cannot access front of empty queue";
    }

    // 使用 if constexpr 避免对 omniback::any 进行 cast
    if constexpr (std::is_same_v<T, omniback::any>) {
      return queue_.front();
    } else {
      return queue_.front().cast<T>();
    }
  }
  
  size_t front_size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (queue_sizes_.size() < queue_.size()){
      return 1;
    } else if (queue_.empty()) {
      TVM_FFI_THROW(tvm::ffi::IndexError) << "Cannot pop from empty queue";
    }
    return queue_sizes_.front();
  }

  /*! \brief Remove front element without returning */
  void pop() {
    TVM_FFI_ICHECK(!queue_.empty()) << "Cannot pop from empty queue";
    queue_.pop_front();
    if (queue_sizes_.size() > queue_.size()){
      real_size_ -= queue_sizes_.front();
      queue_sizes_.pop_front();
    }

    cv_popped_.notify_all();
  }

  /*! \brief Queue size */
  size_t queue_size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return queue_.size();
  }
  size_t size() const{
    std::lock_guard<std::mutex> lock(mutex_);
    const auto qss = queue_sizes_.size();
    const auto qs = queue_.size();
    if (qss == 0){
      return qs;
    } else if (qss == qs){
        return real_size_;
    }
    return real_size_ + qs - qss;
  }

  /*! \brief Check if empty */
  bool empty() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return queue_.empty();
  }

  /*! \brief Clear all elements */
  void clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    queue_.clear();
    queue_sizes_.clear();
    real_size_ = 0;

    // Notify all producers if the queue was full
    cv_popped_.notify_all();
  }

  void set_max_size(size_t new_max_size) {
    std::lock_guard<std::mutex> lock(mutex_);
    max_size_ = new_max_size;
  }

  /*! \brief Get current capacity (0 = unbounded) */
  size_t max_size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return max_size_;
  }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL(
      "omniback.Queue",
      ThreadSafeQueueObj,
      tvm::ffi::Object);

 private:
  template <typename T = omniback::any>
  T unsafe_pop_rm_front() {
    TVM_FFI_ICHECK(!queue_.empty()) << "Cannot pop from empty queue";
    omniback::any value = std::move(queue_.front());
    queue_.pop_front();
    if (queue_sizes_.size() > queue_.size()){
      real_size_ -= queue_sizes_.front();
      queue_sizes_.pop_front();
    }

    cv_popped_.notify_all();

    if constexpr (std::is_same_v<T, omniback::any>) {
      return std::move(value);
    } else {
      return std::move(value).cast<T>();
    }
  }

  template <typename T = omniback::any>
  T unsafe_pop_rm_front(size_t & out_size) {
    TVM_FFI_ICHECK(!queue_.empty()) << "Cannot pop from empty queue";
    omniback::any value = std::move(queue_.front());
    queue_.pop_front();
    if (queue_sizes_.size() > queue_.size()){
      out_size = queue_sizes_.front();
      real_size_ -= out_size;
      queue_sizes_.pop_front();
    }else{
      out_size = 1;
    }

    cv_popped_.notify_all();

    // 使用 if constexpr 避免对 omniback::any 进行 cast
    if constexpr (std::is_same_v<T, omniback::any>) {
      return std::move(value);
    } else {
      return std::move(value).cast<T>();
    }
  }

  mutable std::mutex mutex_;
  std::condition_variable cv_pushed_;
  std::condition_variable cv_popped_; // For bounded queue support
  std::deque<omniback::any> queue_;
  std::deque<size_t> queue_sizes_;
  size_t max_size_{0}; // 0 = unbounded
  size_t real_size_{0}; // 0 = unbounded
};

class ThreadSafeQueueRef : public tvm::ffi::ObjectRef {
 public:

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(
      ThreadSafeQueueRef,
      tvm::ffi::ObjectRef,
      ThreadSafeQueueObj);
};

// Declaration only - definition should be in .cc file
ThreadSafeQueueObj& default_queue(const std::string& tag = "");

} // namespace ffi
} // namespace omniback

#endif // OMNIBACK_FFI_QUEUE_H_