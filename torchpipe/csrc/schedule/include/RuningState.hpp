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
#include <memory>
#include <string>
#include <vector>
#include <mutex>
namespace ipipe {
struct RuningState {
  uint32_t times_of_single_route = 0;
  uint32_t num_runing = 0;
  std::mutex mtx;

  bool skip_waiting_for_batching() {
    std::lock_guard<std::mutex> alock(mtx);
    if (times_of_single_route < 100) {
      return false;
    } else if (times_of_single_route >= 200) {
      times_of_single_route = 100;
      return false;
    } else {
      return true;
    }
  }
};
class RuningStateMonitor {
 public:
  RuningStateMonitor(std::shared_ptr<RuningState> state, std::size_t size_of_num = 1)
      : state_(state), num_add_(size_of_num) {
    std::lock_guard<std::mutex> alock(state_->mtx);
    state_->num_runing += num_add_;

    if (state_->num_runing == 1) {
      state_->times_of_single_route += 1;
    } else {
      state_->times_of_single_route = 0;
    }
  }

  ~RuningStateMonitor() {}

 public:
  // set it to public for compatibility with old code
  void del() {
    assert(state_);
    std::lock_guard<std::mutex> alock(state_->mtx);
    assert(state_->num_runing > 0);
    state_->num_runing -= num_add_;
  }

 private:
  std::shared_ptr<RuningState> state_;
  const uint32_t num_add_;
};
}  // namespace ipipe