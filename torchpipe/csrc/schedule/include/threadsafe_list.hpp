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
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <list>
#include <map>
#include <mutex>
#include <queue>
#include <thread>

namespace ipipe {

template <typename F, typename T>
class ThreadSafeSortList {
 public:
  ThreadSafeSortList() = default;
  ThreadSafeSortList(const ThreadSafeSortList& other) = delete;
  ThreadSafeSortList& operator=(const ThreadSafeSortList& other) = delete;
  struct ScoreValue {
    T data;
    F score;
  };

  void Push(const std::vector<ScoreValue>& new_value) {
    {
      std::lock_guard<std::mutex> lk(mut_);
      for (const auto& item : new_value) data_list_.emplace_back(item);
    }

    data_cond_.notify_one();
  }

  void Push(const T& new_value, F score) {
    {
      std::lock_guard<std::mutex> lk(mut_);
      data_list_.emplace_back(ScoreValue({new_value, score}));
    }

    data_cond_.notify_one();
  }

  // max_num > 0
  std::vector<T> WaitPopRightNow(int time_out, std::size_t max_num) {
    std::unique_lock<std::mutex> lk(mut_);

    auto re = data_cond_.wait_for(lk, std::chrono::milliseconds(time_out),
                                  [this] { return !data_list_.empty(); });
    if (!re) {
      return std::vector<T>();
    }
#ifndef NDEBUG
    const auto original_size = data_list_.size();
#endif
    auto score = data_list_.front().score;
    std::vector<T> value({data_list_.front().data});
    data_list_.pop_front();
    move_nearest(value, score, max_num - 1);
    assert(original_size == value.size() + data_list_.size());
    assert(value.size() <= max_num);
    return value;
  }

  // max_num > 0
  bool WaitUnEmpty(int time_out) {
    std::unique_lock<std::mutex> lk(mut_);

    return data_cond_.wait_for(lk, std::chrono::milliseconds(time_out),
                               [this] { return !data_list_.empty(); });
  }

  // max_num > 0
  std::vector<T> WaitForPop(int time_out, std::size_t max_num) {
    std::unique_lock<std::mutex> lk(mut_);

    data_cond_.wait_for(lk, std::chrono::milliseconds(time_out),
                        [this, max_num] { return max_num <= data_list_.size(); });
    if (data_list_.empty()) {
      return std::vector<T>();
    }
#ifndef NDEBUG
    const auto original_size = data_list_.size();
#endif
    auto front = data_list_.front();
    std::vector<T> value({front.data});
    // value.emplace_back(data_list_.front());
    data_list_.pop_front();
    move_nearest(value, front.score, max_num - 1);
    assert(original_size == value.size() + data_list_.size());
    assert(value.size() <= max_num);
    return value;
  }

  std::vector<T> WaitTimeOut(int time_out, std::size_t max_num, float factor) {
    std::unique_lock<std::mutex> lk(mut_);

    data_cond_.wait_for(lk, std::chrono::milliseconds(time_out),
                        [this, max_num, factor] { return factor * max_num <= data_list_.size(); });
    if (data_list_.empty()) {
      return std::vector<T>();
    }
#ifndef NDEBUG
    auto original_size = data_list_.size();
#endif
    auto front = data_list_.front();
    std::vector<T> value({front.data});
    // value.emplace_back(data_list_.front());
    data_list_.pop_front();
    move_nearest(value, front.score, max_num - 1);
    assert(original_size == value.size() + data_list_.size());
    assert(value.size() <= max_num);
    return value;
  }

  T& front() {
    std::unique_lock<std::mutex> lk(mut_);
    assert(!data_list_.empty());
    return data_list_.front().data;
  }
  //------------------------//------------------------

  void notify_one() { data_cond_.notify_one(); }

  bool empty() const {
    std::lock_guard<std::mutex> lk(mut_);
    return data_list_.empty();
  }

  std::size_t size() const {
    std::lock_guard<std::mutex> lk(mut_);
    return data_list_.size();
  }

 private:
  mutable std::mutex mut_;
  std::list<ScoreValue> data_list_;
  std::condition_variable data_cond_;

  void move_nearest(std::vector<T>& value, float score, std::size_t max_num) {
    std::vector<decltype(data_list_.begin())> iters;
    for (auto iter = data_list_.begin(); iter != data_list_.end(); ++iter) {
      iters.emplace_back(iter);
    }
    std::stable_sort(iters.begin(), iters.end(), [score](auto a, auto b) {
      return fabs(score - a->score) < fabs(score - b->score);
    });
    const auto len = std::min(max_num, iters.size());

    for (std::size_t i = 0; i < len; ++i) {
      value.emplace_back(iters[i]->data);
      data_list_.erase(iters[i]);
    }
  }
};
}  // namespace ipipe