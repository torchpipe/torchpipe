/*!
 * \file omniback/ffi/queue.h
 * \brief Runtime queue container type. Thread-safe.
 */

#ifndef OMNIBACK_FFI_QUEUE_H_
#define OMNIBACK_FFI_QUEUE_H_

#include <tvm/ffi/object.h>
#include <tvm/ffi/memory.h>
#include <tvm/ffi/any.h>
#include <tvm/ffi/optional.h>

#include <condition_variable>
#include <deque>
#include <mutex>
#include <chrono>
#include <utility>

namespace omniback {
namespace ffi {

/*!
 * \brief FFIQueueObj stores queue elements, supports thread-safe push/pop.
 */
class FFIQueueObj : public tvm::ffi::Object {
 public:
  FFIQueueObj() = default;
  // https://github.com/apache/tvm-ffi/blob/f7e09d6a96b54554190bae0d7ba9ff7a6e9a109e/include/tvm/ffi/object.h#L175
  static constexpr bool _type_mutable = true;
  
  void push(const tvm::ffi::Any& value) {
    std::lock_guard<std::mutex> lock(mutex_);
    queue_.push_back(value);
    cv_not_empty_.notify_one();
  }

  void push(tvm::ffi::Any&& value) {
    std::lock_guard<std::mutex> lock(mutex_);
    queue_.push_back(std::move(value));
    cv_not_empty_.notify_one();
  }

  tvm::ffi::Optional<tvm::ffi::Any> pop(
      std::chrono::milliseconds timeout = std::chrono::milliseconds::max()) {
    std::unique_lock<std::mutex> lock(mutex_);

    if (timeout == std::chrono::milliseconds::max()) {
      cv_not_empty_.wait(lock, [this]() { return !queue_.empty(); });
    } else {
      if (!cv_not_empty_.wait_for(
              lock, timeout, [this]() { return !queue_.empty(); })) {
        return tvm::ffi::Optional<tvm::ffi::Any>();
      }
    }

    tvm::ffi::Any value = std::move(queue_.front());
    queue_.pop_front();
    return tvm::ffi::Optional<tvm::ffi::Any>(std::move(value));
  }

  tvm::ffi::Optional<tvm::ffi::Any> try_pop() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (queue_.empty()) {
      return tvm::ffi::Optional<tvm::ffi::Any>();
    }
    tvm::ffi::Any value = std::move(queue_.front());
    queue_.pop_front();
    return tvm::ffi::Optional<tvm::ffi::Any>(std::move(value));
  }

  tvm::ffi::Optional<tvm::ffi::Any> front() const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (queue_.empty()) {
      return tvm::ffi::Optional<tvm::ffi::Any>();
    }
    return tvm::ffi::Optional<tvm::ffi::Any>(queue_.front());
  }

  size_t size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return queue_.size();
  }

  bool empty() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return queue_.empty();
  }

  void clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    queue_ = std::deque<tvm::ffi::Any>();
  }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL(
      "omniback.FFIQueue",
      FFIQueueObj,
      tvm::ffi::Object);

 private:
  mutable std::mutex mutex_;
  std::condition_variable cv_not_empty_;
  std::deque<tvm::ffi::Any> queue_;
};

/*!
 * \brief Reference class for FFIQueue.
 */
class FFIQueue : public tvm::ffi::ObjectRef {
 public:
  // explicit FFIQueue(tvm::ffi::String tag){
  //   data_ = 
  // }

   void push(const tvm::ffi::Any& value) {
    GetMutableObj()->push(value);
  }

  void push(tvm::ffi::Any&& value) {
    GetMutableObj()->push(std::move(value));
  }

  tvm::ffi::Optional<tvm::ffi::Any> pop(
      std::chrono::milliseconds timeout = std::chrono::milliseconds::max()) {
    return GetMutableObj()->pop(timeout);
  }

  tvm::ffi::Optional<tvm::ffi::Any> try_pop() {
    return GetMutableObj()->try_pop();
  }

  void clear() {
    GetMutableObj()->clear();
  }

  // const版本：只读操作
  tvm::ffi::Optional<tvm::ffi::Any> front() const {
    return GetImmutableObj()->front();
  }

  size_t size() const {
    return GetImmutableObj()->size();
  }

  bool empty() const {
    return GetImmutableObj()->empty();
  }

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(
      FFIQueue,
      tvm::ffi::ObjectRef,
      FFIQueueObj);

 private:
  FFIQueueObj* GetMutableObj() {
    return static_cast<FFIQueueObj*>(data_.get());
  }

  const FFIQueueObj* GetImmutableObj() const {
    return static_cast<const FFIQueueObj*>(data_.get());
  }
};



} // namespace ffi
} // namespace omniback

#endif // OMNIBACK_FFI_QUEUE_H_