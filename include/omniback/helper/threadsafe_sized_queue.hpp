// Copyright 2021-2025 NetEase.
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
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>
#include "omniback/helper/sized_queue.hpp"

// #include "base_logging.hpp"
namespace omniback {

template <typename T>
class ThreadSafeSizedQueue {
 public:
  ThreadSafeSizedQueue() = default;
  ThreadSafeSizedQueue(const ThreadSafeSizedQueue& other) = delete;
  ThreadSafeSizedQueue& operator=(const ThreadSafeSizedQueue& other) = delete;

  void push(const T& new_value, size_t size) {
    {
      std::lock_guard<std::mutex> lk(mut_);
      data_queue_.push(new_value, size);
    }

    data_cond_.notify_all();
  }

  void notify_all() {
    data_cond_.notify_all();
    // https://wanghenshui.github.io/2019/08/23/notify-one-pred
  }

  // bool wait(int time_out) {
  //     std::unique_lock<std::mutex> lk(mut_);
  //     return std::cv_status::timeout !=
  //            data_cond_.wait_for(lk, std::chrono::milliseconds(time_out));
  // }

  // void PushIfEmpty(const T& new_value, size_t size) {
  //   {
  //     std::unique_lock<std::mutex> lk(mut_);
  //     poped_cond_.wait(lk, [this] { return data_queue_.empty(); });
  //     data_queue_.push(new_value, size);
  //   }

  //   data_cond_.notify_all();
  // }

  void push(const std::vector<T>& new_value, const std::vector<size_t>& sizes) {
    std::lock_guard<std::mutex> lk(mut_);
    for (size_t i = 0; i < new_value.size(); ++i)
      data_queue_.push(new_value[i], sizes[i]);

    data_cond_.notify_all();
  }

  void push(
      const std::vector<T>& new_value,
      const std::function<int(const T&)>& req_size_func) {
    std::lock_guard<std::mutex> lk(mut_);
    for (size_t i = 0; i < new_value.size(); ++i)
      data_queue_.push(new_value[i], req_size_func(new_value[i]));

    data_cond_.notify_all();
  }

  void notify_one() {
    data_cond_.notify_one();
    // https://wanghenshui.github.io/2019/08/23/notify-one-pred
  }

  void wait_pop(T& value) {
    {
      std::unique_lock<std::mutex> lk(mut_);
      data_cond_.wait(lk, [this] { return !data_queue_.empty(); });
      value = data_queue_.front();
      data_queue_.pop();
    }
  }

  T pop() {
    {
      std::unique_lock<std::mutex> lk(mut_);
      data_cond_.wait(lk, [this] { return !data_queue_.empty(); });
      auto value = data_queue_.front();
      data_queue_.pop();
      return value;
    }
  }

  T& front() {
    std::unique_lock<std::mutex> lk(mut_);
    assert(!data_queue_.empty());
    return data_queue_.front();
  }

  T WaitPop(std::function<bool(const T&)> check = [](const T&) {
    return true;
  }) {
    std::unique_lock<std::mutex> lk(mut_);
    data_cond_.wait(lk, [this, check] {
      return !data_queue_.empty() && check(data_queue_.front());
    });
    T res = data_queue_.front();
    data_queue_.pop();

    poped_cond_.notify_all();
    return res;
  }

  bool wait_pop(T& value, int time_out) {
    std::unique_lock<std::mutex> lk(mut_);
    auto re =
        data_cond_.wait_for(lk, std::chrono::milliseconds(time_out), [this] {
          return !data_queue_.empty();
        });
    if (!re)
      return false;
    value = data_queue_.front();
    data_queue_.pop();

    poped_cond_.notify_all();
    return true;
  }

  bool WaitForPop(T& value, int time_out, std::function<bool(const T&)> check) {
    std::unique_lock<std::mutex> lk(mut_);
    auto re = data_cond_.wait_for(
        lk, std::chrono::milliseconds(time_out), [this, check] {
          return !data_queue_.empty() && check(data_queue_.front());
        });
    if (!re)
      return false;
    value = data_queue_.front();
    data_queue_.pop();

    poped_cond_.notify_all();
    return true;
  }

  bool WaitForPopWithSize(
      T& value,
      int time_out,
      std::function<bool(std::size_t)> check) {
    std::unique_lock<std::mutex> lk(mut_);
    auto re = data_cond_.wait_for(
        lk, std::chrono::milliseconds(time_out), [this, check] {
          return !data_queue_.empty() && check(data_queue_.front_size());
        });
    if (!re)
      return false;
    value = data_queue_.front();
    data_queue_.pop();

    poped_cond_.notify_all();
    return true;
  }

  bool WaitForPopWithConditionAndStatus(
      T& value,
      int time_out,
      std::function<bool(std::size_t)> check) {
    std::unique_lock<std::mutex> lk(mut_);

    if (data_queue_.empty() || !check(data_queue_.front_size())) {
      num_waiting_++;
      // SPDLOG_INFO("WaitForPopWithConditionAndStatus: num_waiting_ =
      // {}", num_waiting_); lk.unlock();
      waiting_cond_.notify_all();
      // lk.lock();
      auto re = data_cond_.wait_for(
          lk, std::chrono::milliseconds(time_out), [this, check] {
            return !data_queue_.empty() && check(data_queue_.front_size());
          });
      num_waiting_--;
      // SPDLOG_INFO("WaitForPopWithConditionAndStatus finish:
      // num_waiting_ = {}", num_waiting_);
      if (!re)
        return false;
    }

    value = data_queue_.front();
    data_queue_.pop();

    lk.unlock();
    poped_cond_.notify_all();
    return true;
  }

  bool WaitForWaiting(int time_out) {
    std::unique_lock<std::mutex> lk(mut_);

    auto re =
        waiting_cond_.wait_for(lk, std::chrono::milliseconds(time_out), [this] {
          return num_waiting_ > 0;
        });
    if (!re)
      return false;
    // SPDLOG_INFO("WaitForWaiting: num_waiting_ = {}", num_waiting_);
    return true;
  }

  // bool wait_for(int time_out) {
  //     std::unique_lock<std::mutex> lk(mut_);
  //     return data_cond_.wait_for(lk,
  //     std::chrono::milliseconds(int(time_out)),
  //                                [this] { return !data_queue_.empty(); });
  // }

  bool wait_for(int time_out) {
    std::unique_lock<std::mutex> lk(mut_);
    return data_cond_.wait_for(
        lk, std::chrono::milliseconds((time_out)), [this] {
          return !data_queue_.empty();
        });
  }

  void WaitLessThan(int num, int time_out) {
    std::unique_lock<std::mutex> lk(mut_);
    data_cond_.wait_for(
        lk, std::chrono::milliseconds(int(time_out)), [this, num] {
          return data_queue_.size() < num;
        });
  }

  bool empty() const {
    std::lock_guard<std::mutex> lk(mut_);
    return data_queue_.empty();
  }

  std::size_t size() const {
    std::lock_guard<std::mutex> lk(mut_);
    return data_queue_.size();
  }

  std::size_t front_size() const {
    std::lock_guard<std::mutex> lk(mut_);
    return data_queue_.front_size();
  }

 private:
  mutable std::mutex mut_;
  SizedQueue<T> data_queue_;
  std::condition_variable data_cond_;
  std::condition_variable poped_cond_;
  uint32_t num_waiting_{0};
  std::condition_variable waiting_cond_;
};

} // namespace omniback