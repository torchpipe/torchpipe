// Copyright 2021-2024 NetEase.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <atomic>
#include <cassert>
#include <condition_variable>
#include <functional>
#include <iostream>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <unordered_map>
#include <vector>
#include "time_utils.hpp"

namespace ipipe {
/// @brief  代表内置事件的键值。参见@ref SimpleEvents
constexpr auto TASK_EVENT_KEY = "event";
/**
 * @brief 事件，类似于python的Event，用于线程间通信.
 * @see TASK_EVENT_KEY
 *
 */
class SimpleEvents {
 public:
  /**
   * @brief
   * @param num  引用计数目标值。引用计数原始为0，每被通知一次，增加1.
   *
   */
  SimpleEvents(uint32_t num = 1);

  /**
   * @brief
   * 注意，任务并未完成，请自行防止出现潜在的多线程问题。
   */
  void WaitDanger() {
    std::unique_lock<std::mutex> lk(mut);

    data_cond.wait(lk, [this] { return (num_task == ref_count) || eptr_; });
    if (eptr_) {
      std::rethrow_exception(eptr_);
    }
  }

  bool WaitFinish(uint32_t timeout_ms) {
    std::unique_lock<std::mutex> lk(mut);

    return data_cond.wait_for(lk, std::chrono::milliseconds(timeout_ms),
                              [this] { return (num_task == ref_count); });  //
  }

  /// true: 引用计数等小于引用计数目标值时。
  bool valid() {
    std::lock_guard<std::mutex> lk(mut);
    return num_task > ref_count;
  }

  /// @brief 阻塞，以条件引用计数等于引用计数目标值进行监听。
  /// @return std::exception_ptr 异常
  std::exception_ptr WaitAndGetExcept() {
    std::unique_lock<std::mutex> lk(mut);

    data_cond.wait(lk, [this] { return (num_task == ref_count); });  //
    return eptr_;
  }

  void Wait() {
    std::unique_lock<std::mutex> lk(mut);

    data_cond.wait(lk, [this] { return (num_task == ref_count); });  //
    if (eptr_) {
      std::rethrow_exception(eptr_);
    }
  }

  bool Wait(uint32_t timeout_ms) {
    std::unique_lock<std::mutex> lk(mut);

    bool done = data_cond.wait_for(lk, std::chrono::milliseconds(timeout_ms),
                                   [this] { return (num_task == ref_count); });  //
    if (eptr_) {
      std::rethrow_exception(eptr_);
    }
    return done;
  }

  bool WaitDanger(uint32_t timeout_ms) {
    std::unique_lock<std::mutex> lk(mut);

    bool done = data_cond.wait_for(lk, std::chrono::milliseconds(timeout_ms),
                                   [this] { return (num_task == ref_count) || eptr_; });  //
    if (eptr_) {
      std::rethrow_exception(eptr_);
    }
    return done;
  }

  /**
   * @brief 通知一个监听者。
   * @remark 具体步骤为：
   * 1. 引用计数加一；
   * 2. 如果引用计数等于引用计数目标值，并且设置了回调函数，
   *    执行回调并清除设置的回调函数；
   * 3. 无条件唤醒一个阻塞的监听者。
   */
  void notify_one() {
    // 放弃dict所有权
    {
      // do not remove this lock even if ref_count is atomic
      std::lock_guard<std::mutex> lk(mut);
      ref_add();
    }
    try_callback();

    data_cond.notify_one();
  }

  void notify_one(int num) {
    {
      // do not remove this lock even if ref_count is atomic
      std::lock_guard<std::mutex> lk(mut);
      ref_add(num);
    }
    try_callback();

    data_cond.notify_one();
  }
  /**
   * @brief 通知所有监听者。
   * @ref SimpleEvents::notify_one
   */
  void notify_all() {
    {
      // do not remove this lock even if ref_count is atomic
      std::lock_guard<std::mutex> lk(mut);
      ref_add();
    }
    try_callback();

    data_cond.notify_all();
  }

  void notify_all(int num) {
    {
      // do not remove this lock even if ref_count is atomic
      std::lock_guard<std::mutex> lk(mut);
      ref_add(num);
    }
    try_callback();

    data_cond.notify_all();
  }

  /// @brief  是否被设置了异常。
  bool has_exception() {
    std::lock_guard<std::mutex> lk(mut);
    return eptr_ != nullptr;
  }

  /**
   * @brief 设置异常并通知。
   * @remarks 具体步骤为：
   * 1. 引用计数加一；
   * 2. 如果不含有异常， 设置异常 eptr
   * 3. 如果引用计数等于引用计数目标值，并且设置了回调函数，
   *    执行回调并清除设置的回调函数；
   * 4. 无条件唤醒一个阻塞的监听者。
   */
  void set_exception_and_notify_one(std::exception_ptr eptr) {
    {
      std::lock_guard<std::mutex> lk(mut);

      if (!eptr_) eptr_ = eptr;
      ref_add();
    }
    try_callback();

    data_cond.notify_one();
  }
  void ref_add() {
    if (ref_count < num_task)
      ref_count++;
    else {
      assert(false);
    }
  }

  void ref_add(int num) {
    if (ref_count < num_task)
      ref_count += num;
    else {
      assert(false);
    }
    assert(ref_count <= num_task);
  }

  /**
   * @brief 清理并返回被设置的异常
   */
  std::exception_ptr reset_exception() {
    std::lock_guard<std::mutex> lk(mut);
    auto re = eptr_;
    eptr_ = nullptr;
    return re;
  }

  std::exception_ptr get_exception() {
    std::lock_guard<std::mutex> lk(mut);

    return eptr_;
  }

  void try_throw() {
    std::lock_guard<std::mutex> lk(mut);

    if (eptr_) {
      std::rethrow_exception(eptr_);
    }
  }

  void task_add(int num) {
    std::lock_guard<std::mutex> lk(mut);
    num_task += num;
  }
  void try_callback() {
    bool should_try = false;
    {
      std::lock_guard<std::mutex> lk(mut);
      should_try = (ref_count == num_task);
    }

    if (should_try) {
      while (!callbacks_.empty()) {
        callbacks_.back()();    // Execute the last callback
        callbacks_.pop_back();  // Remove the last callback
      }
      while (!latest_callbacks_.empty()) {
        latest_callbacks_.back()();    // Execute the last callback
        latest_callbacks_.pop_back();  // Remove the last callback
      }
    }
  }

  /// 参见 @ref SimpleEvents::set_exception_and_notify_one
  void set_exception_and_notify_all(std::exception_ptr eptr) {
    {
      std::lock_guard<std::mutex> lk(mut);

      if (!eptr_) eptr_ = eptr;
      ref_add();
    }
    try_callback();

    data_cond.notify_all();
  }

  /// 设置回调函数
  void add_const_callback(std::function<void()> callback) {
    std::lock_guard<std::mutex> lk(mut);
    // assert(!callback_);
    callbacks_.emplace_back(callback);
  }

  void set_callback(std::function<void()> callback) {
    std::lock_guard<std::mutex> lk(mut);
    // assert(!callback_);
    { latest_callbacks_.push_back(callback); }
  }

  void set_final_callback(std::function<void()> callback) {
    std::lock_guard<std::mutex> lk(mut);
    // assert(!callback_);

    latest_callbacks_.push_back(callback);
    if (latest_callbacks_.size() > 1) {
      latest_callbacks_.pop_back();
      throw std::runtime_error(
          "The callback stack is not empty. Consider using set_callback instead.");
    }
  }

  /// 获得从构造到现在经过的时间（单位：毫秒）。
  float time_passed();

  ~SimpleEvents() {
    // std::lock_guard<std::mutex> lk(mut);
    // if (ref_count != num_task) {
    // }
  };

 private:
  std::mutex mut;  // mutable
  std::condition_variable data_cond;
  uint32_t ref_count = 0;
  uint32_t num_task;
  std::vector<std::function<void()>> callbacks_;
  std::vector<std::function<void()>> latest_callbacks_;

  std::exception_ptr eptr_;
  std::chrono::steady_clock::time_point starttime_;
};

static inline std::shared_ptr<SimpleEvents> make_event(uint32_t num = 1) {
  return std::make_shared<SimpleEvents>(num);
}

#ifndef DOXYGEN_SHOULD_SKIP_THIS
class SharedSimpleEvents {
 public:
  SharedSimpleEvents(uint32_t num = 1) { _data = std::make_shared<SimpleEvents>(num); }

  SharedSimpleEvents(std::shared_ptr<SimpleEvents> input_data) { _data = input_data; }

  void Wait() { _data->Wait(); }

  void notify_one() { _data->notify_one(); }

  std::shared_ptr<SimpleEvents> data() { return _data; }

  operator std::shared_ptr<SimpleEvents>() { return _data; }

 private:
  std::shared_ptr<SimpleEvents> _data;
};
#endif

}  // namespace ipipe