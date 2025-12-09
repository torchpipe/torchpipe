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
#include <functional>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

namespace omniback {

template <typename T>
class ThreadSafeQueue {
 public:
  ThreadSafeQueue() = default;
  ThreadSafeQueue(const ThreadSafeQueue& other) = delete;
  ThreadSafeQueue& operator=(const ThreadSafeQueue& other) = delete;

  void push(const T& new_value) {
    {
      std::lock_guard<std::mutex> lk(mut_);
      data_queue_.push(new_value);
    }

    data_cond_.notify_all();
  }

  void push_and_notify_one(const T& new_value) {
    {
      std::lock_guard<std::mutex> lk(mut_);
      data_queue_.push(new_value);
    }

    data_cond_.notify_one();
  }

  // void PushIfEmpty(const T& new_value) {
  //   {
  //     std::unique_lock<std::mutex> lk(mut_);
  //     poped_cond_.wait(lk, [this] { return data_queue_.empty(); });
  //     data_queue_.push(new_value);
  //   }

  //   data_cond_.notify_all();
  // }

  void push(const std::vector<T>& new_value) {
    {
      std::lock_guard<std::mutex> lk(mut_);
      for (const auto& item : new_value)
        data_queue_.push(item);
    }

    data_cond_.notify_all();
  }

  void notify_one() {
    data_cond_.notify_one();
    // https://wanghenshui.github.io/2019/08/23/notify-one-pred
  }

  void notify_all() {
    data_cond_.notify_all();
  }

  void WaitPop(T& value) {
    {
      std::unique_lock<std::mutex> lk(mut_);
      data_cond_.wait(lk, [this] { return !data_queue_.empty(); });
      value = data_queue_.front();
      data_queue_.pop();
    }

    poped_cond_.notify_all();
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
    {
      std::unique_lock<std::mutex> lk(mut_);
      auto re =
          data_cond_.wait_for(lk, std::chrono::milliseconds(time_out), [this] {
            return !data_queue_.empty();
          });
      if (!re)
        return false;
      std::swap(value, data_queue_.front());
      // value = data_queue_.front();
      data_queue_.pop();
    }

    poped_cond_.notify_all();
    return true;
  }

  bool WaitEmpty(int time_out) {
    std::unique_lock<std::mutex> lk(mut_);
    auto re =
        poped_cond_.wait_for(lk, std::chrono::milliseconds(time_out), [this] {
          return data_queue_.empty();
        });
    return re;
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

  std::vector<T> PopAll() {
    std::unique_lock<std::mutex> lk(mut_);
    std::vector<T> result;
    while (!data_queue_.empty()) {
      result.push_back(data_queue_.front());
      data_queue_.pop();
    }
    return result;
  }

  void Wait(int time_out) {
    std::unique_lock<std::mutex> lk(mut_);
    data_cond_.wait_for(lk, std::chrono::milliseconds(int(time_out)), [this] {
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

 private:
  mutable std::mutex mut_;
  std::queue<T> data_queue_;
  std::condition_variable data_cond_;
  std::condition_variable poped_cond_;
};

template <typename T>
class SingleElementQueue {
 public:
  SingleElementQueue() : has_data(false) {}

  // 推送数据（非阻塞）
  bool push(const T& value) {
    std::lock_guard<std::mutex> lock(mtx);

    if (has_data) {
      return false; // 队列已满
    }

    data = value;
    has_data = true;

    // 通知等待的消费者
    cv.notify_one();
    return true;
  }

  // 非阻塞弹出
  std::optional<T> try_pop() {
    std::lock_guard<std::mutex> lock(mtx);

    if (!has_data) {
      return std::nullopt;
    }

    T value = data;
    has_data = false;
    return value;
  }

  // 阻塞弹出（无限等待）
  T pop() {
    std::unique_lock<std::mutex> lock(mtx);

    // 等待直到有数据可用
    cv.wait(lock, [this] { return has_data; });

    T value = data;
    has_data = false;
    return value;
  }

  // 带超时的阻塞弹出
  template <typename Rep, typename Period>
  std::optional<T> pop_timeout(
      const std::chrono::duration<Rep, Period>& timeout) {
    std::unique_lock<std::mutex> lock(mtx);

    // 等待数据或超时
    if (!cv.wait_for(lock, timeout, [this] { return has_data; })) {
      return std::nullopt; // 超时
    }

    T value = data;
    has_data = false;
    return value;
  }

  bool wait_pop(T& out, int timeout_ms) {
    std::unique_lock<std::mutex> lock(mtx);
    // 有限等待
    auto status =
        cv.wait_for(lock, std::chrono::milliseconds(timeout_ms), [this] {
          return has_data;
        });

    // 超时检查
    if (status) {
      std::swap(out, data);
      has_data = false;
      return true;
    }

    return false;
  }

 private:
  T data;
  bool has_data; // 数据存在标志
  std::mutex mtx;
  std::condition_variable cv;
};

} // namespace omniback