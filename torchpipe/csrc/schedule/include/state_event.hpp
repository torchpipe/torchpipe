// Copyright 2021-2023 NetEase.
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
#include <unordered_map>
#include <vector>
// #include "time_utils.hpp"

namespace ipipe {
class StateEvents {
 public:
  enum struct State { invalid, full, empty, half };

  StateEvents(uint32_t num = 1) : num_task_(num) {}

  bool WaitUnFull(int time_out, State& out) {
    std::unique_lock<std::mutex> lk(mut);
    assert(state_ != State::invalid);
    auto re = data_cond_.wait_for(lk, std::chrono::milliseconds(time_out), [this] {
      return (state_ == State::empty) || (state_ == State::half);
    });  //
    if (!re) return false;

    out = state_;
    return true;
  }

  void add() {
    {
      std::lock_guard<std::mutex> lk(mut);
      if (++num_running_ != num_task_) {
        state_ = State::half;
      } else {
        state_ = State::full;
      }
    }

    data_cond_.notify_one();
  }

  void del() {
    {
      std::lock_guard<std::mutex> lk(mut);
      if (--num_running_ != 0) {
        state_ = State::half;
      } else {
        state_ = State::empty;
      }
    }
    data_cond_.notify_one();
  }

 private:
  std::multiset<int> batch_sizes_;
  State state_ = State::empty;
  std::mutex mut;
  std::condition_variable data_cond_;
  const uint32_t num_task_;
  std::size_t num_running_;
};
struct StateEventsGuard {
  StateEventsGuard(StateEvents* pstate) {
    if (pstate) {
      pstate_ = pstate;
      pstate_->add();
    }
  }
  ~StateEventsGuard() {
    if (pstate_) pstate_->del();
  }
  StateEvents* pstate_ = nullptr;
};
}  // namespace ipipe