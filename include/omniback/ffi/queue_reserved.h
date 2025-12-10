/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file omniback/ffi/queue.h
 * \brief Runtime queue container type. Thread-safe. Supports batch size for
 * each element.
 */

#ifndef OMNIBACK_FFI_QUEUE_H_
#define OMNIBACK_FFI_QUEUE_H_

#include <tvm/ffi/object.h>
#include <tvm/ffi/memory.h>
#include <tvm/ffi/any.h>
#include <tvm/ffi/container/container_details.h>
#include <tvm/ffi/container/tuple.h>

#include <condition_variable>
#include <deque>
#include <mutex>
#include <optional>
#include <utility>

namespace omniback {
namespace ffi {

template <typename... Args>
tvm::ffi::Tuple<std::decay_t<Args>...> make_tuple(Args&&... args) {
    return tvm::ffi::Tuple<std::decay_t<Args>...>(std::forward<Args>(args)...);
}

/*!
 * \brief SizedQueueObj stores queue elements and their batch sizes, supports
 * thread-safe push/pop. Not a template (template handled in wrapper). Entry =
 * tvm::ffi::Tuple<tvm::ffi::Any, int32_t>
 */
class SizedQueueObj : public tvm::ffi::Object {
 public:
  using Entry = tvm::ffi::Tuple<tvm::ffi::Any, int32_t>;

  SizedQueueObj() = default;

  void push(const tvm::ffi::Any& value, int32_t batch_size = 1) {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      // 使用emplace_back避免临时Entry对象构造
      queue_.emplace_back(value, batch_size);
    }
    cv_not_empty_.notify_one();
  }

  std::optional<Entry> pop() {
    std::unique_lock<std::mutex> lock(mutex_);
    // 显式捕获this，避免使用[&]捕获所有变量
    cv_not_empty_.wait(lock, [this]() { return !queue_.empty(); });
    Entry entry = std::move(queue_.front());
    queue_.pop_front();
    return entry;
  }

  std::optional<Entry> try_pop() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (queue_.empty())
      return std::nullopt;
    Entry entry = std::move(queue_.front());
    queue_.pop_front();
    return entry;
  }

  std::optional<Entry> front() const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (queue_.empty())
      return std::nullopt;
    return queue_.front(); // 返回拷贝，不移动
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
    // 使用swap释放内存，而不是简单的clear()
    std::deque<Entry>().swap(queue_);
  }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL(
      "omniback.SizedQueue",
      SizedQueueObj,
      tvm::ffi::Object);

 private:
  mutable std::mutex mutex_;
  std::condition_variable cv_not_empty_;
  std::deque<Entry> queue_;
};

/*!
 * \brief Template reference class for SizedQueue.
 * \tparam T The value type, must be FFI-storage-enabled!
 */
template <
    typename T,
    typename =
        typename std::enable_if_t<tvm::ffi::details::storage_enabled_v<T>>>
class SizedQueue : public tvm::ffi::ObjectRef {
 public:
  using Entry = typename SizedQueueObj::Entry;
  /*! \brief Construct an empty queue */
  SizedQueue() {
    data_ = tvm::ffi::make_object<SizedQueueObj>();
  }
  /*! \brief Construct from ObjectPtr (for FFI) */
  explicit SizedQueue(tvm::ffi::ObjectPtr<tvm::ffi::Object> ptr)
      : tvm::ffi::ObjectRef(ptr) {}

  /*! \brief Threadsafe push (waits for space if necessary). Returns void. */
  void push(const T& value, int32_t batch_size = 1) const {
    GetObj()->push(tvm::ffi::Any(value), batch_size);
  }

  /*! \brief Threadsafe pop (waits if empty). Returns tuple<T, batch_size>. */
  tvm::ffi::Tuple<T, int32_t> pop() const {
    auto opt = GetObj()->pop();
    if (!opt.has_value()) {
      throw std::runtime_error("pop failed (queue closed?)");
    }
    return make_tuple(
        std::move(opt->get<0>().template cast<T>()), opt->get<1>());
  }

  /*! \brief Try pop (does not wait), returns nullopt if empty. */
  std::optional<tvm::ffi::Tuple<T, int32_t>> try_pop() const {
    auto opt = GetObj()->try_pop();
    if (!opt.has_value())
      return std::nullopt;
    return make_tuple(
        std::move(opt->get<0>().template cast<T>()), opt->get<1>());
  }

  /*! \brief View the front of the queue, does not remove. */
  std::optional<tvm::ffi::Tuple<T, int32_t>> front() const {
    auto opt = GetObj()->front();
    if (!opt.has_value())
      return std::nullopt;
    // 注意：这里使用拷贝而非移动，因为队列中元素需要保留
    return make_tuple(
        opt->get<0>().template cast<T>(), opt->get<1>());
  }

  size_t size() const {
    return GetObj()->size();
  }

  bool empty() const {
    return GetObj()->empty();
  }

  void clear() const {
    GetObj()->clear();
  }

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(
      SizedQueue,
      tvm::ffi::ObjectRef,
      SizedQueueObj);

 private:
  SizedQueueObj* GetObj() const {
    return static_cast<SizedQueueObj*>(data_.get());
  }
};

} // namespace ffi
} // namespace omniback

#endif // OMNIBACK_FFI_QUEUE_H_