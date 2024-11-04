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
#include "base_logging.hpp"
#include <memory>
#include <string>
#include <vector>
#include "Backend.hpp"
#include "dict.hpp"
#include "event.hpp"
#include "MultipleInstances.hpp"
#include "params.hpp"
#include "reflect.h"
#include "threadsafe_queue_sized.hpp"
#include "time_utils.hpp"
#include "RangeMerger.hpp"
#include "RuningState.hpp"
#include <cassert>
namespace ipipe {

class RequestStates {
 public:
  struct RequestState {
   public:
    int iter_index = 0;
    bool wait_for_schedule = true;
  };
  bool wait_decode_ready(int time_out) {
    std::unique_lock<std::mutex> lock(mtx_);
    return cv_.wait_for(lock, std::chrono::milliseconds(time_out), [this]() {
      for (auto iter = request_states_.begin(); iter != request_states_.end(); ++iter) {
        if (iter->second.iter_index >= 1 && !iter->second.wait_for_schedule) {
          return false;
        }
      }
      return true;
    });
  }

  void remove(const std::string& request_id) {
    {
      std::lock_guard<std::mutex> lock(mtx_);
      request_states_.erase(request_id);
    }
    cv_.notify_all();
  }

  void set_wait(const std::string& request_id) {
    std::lock_guard<std::mutex> lock(mtx_);
    auto iter = request_states_.find(request_id);
    if (iter != request_states_.end()) {
      iter->second.wait_for_schedule = true;

    } else {
      // (*request_states_)[request_id] = RequestState({0, true});
      request_states_.emplace(request_id, RequestState({0, true}));
    }
    cv_.notify_all();
  }

  void set_unwait(const std::string& request_id) {
    std::lock_guard<std::mutex> lock(mtx_);
    auto iter = request_states_.find(request_id);
    if (iter != request_states_.end()) {
      iter->second.wait_for_schedule = false;
      iter->second.iter_index += 1;
      // cv_.notify_all();
    }
  }
  // void notify_all() {
  //   std::lock_guard<std::mutex> lock(mtx_);
  //   cv_.notify_all();
  // }

 private:
  std::unordered_map<std::string, RequestState> request_states_;
  mutable std::mutex mtx_;
  std::condition_variable cv_;
};
// # cal_request_size_method = "AddRequestSizeTensor"

class Batching : public Backend {
 public:
  bool init(const std::unordered_map<std::string, std::string>& config, dict dict_config);
  /**
   * @return UINT32_MAX.
   */
  virtual uint32_t max() const { return UINT32_MAX; };

  void forward(const std::vector<dict>& raw_inputs) {
    if (cal_request_size_method_) {
      // if (!bThreadInited_.load()) {
      //   SPDLOG_ERROR("cal_request_size_method_ is not supported when no batching needed");
      //   abort();
      // }
      for (const auto& item : raw_inputs) cal_request_size_method_->forward({item});
    }
    std::vector<std::shared_ptr<SimpleEvents>> events;  // 注意，
    // 事件需要提前准备好，不可运行时从map获得，容易造成多线程问题

    for (auto raw_input : raw_inputs) {
      std::shared_ptr<RuningStateMonitor> guard_state =
          std::make_shared<RuningStateMonitor>(runing_state_, 1);
      assert(guard_state);
      auto& map_data = *raw_input;
      map_data.erase(TASK_RESULT_KEY);

      auto iter = raw_input->find(TASK_EVENT_KEY);
      if (iter == raw_input->end()) {
        auto event = make_event();
        events.emplace_back(event);
        event->add_const_callback([guard_state]() { guard_state->del(); });
        if (cal_request_size_method_) {
          auto* data = raw_input.get();
          event->add_const_callback([data]() { data->erase(TASK_REQUEST_SIZE_KEY); });
        }
        map_data[TASK_EVENT_KEY] = event;
      } else {
        events.emplace_back(nullptr);

        std::shared_ptr<SimpleEvents> ev = any_cast<std::shared_ptr<SimpleEvents>>(iter->second);
        ev->add_const_callback([guard_state]() { guard_state->del(); });
        if (cal_request_size_method_) {
          auto* data = raw_input.get();
          ev->add_const_callback([data]() { data->erase(TASK_REQUEST_SIZE_KEY); });
        }
      }
    }

    assert(events.size() == raw_inputs.size());

    {
      // 注意：资源所有权问题， 从此刻起 对 raw_input 没有读写权限，
      // 除非event通知

      if (!bThreadInited_.load()) {
        for (auto raw_input : raw_inputs) {
          backend_->forward({raw_input});  // 异步调用, bs=1
        }
      } else {
        std::vector<size_t> sizes;
        for (const auto& item : raw_inputs) {
          const auto item_size = get_request_size(item);
          // SPDLOG_DEBUG("item_size={} max_batch_size={}", item_size, max_batch_size_);
          IPIPE_ASSERT(item_size <= max_batch_size_);
          sizes.push_back(item_size);
        }
        input_queue_.Push(raw_inputs, sizes);  // todo 限制送入的不能超过最大值
      }
    }
    if (request_states_) {
      for (const auto& request : raw_inputs) {
        auto iter = request->find("request_id");

        std::string* request_id = any_cast<std::string>(&iter->second);
        request_states_->set_wait(*request_id);
      }
    }

    for (std::size_t i = 0; i < raw_inputs.size(); ++i) {
      // 当阻塞式调用时 todo  非阻塞调用
      if (events[i]) {
        events[i]->Wait();
        // 重新获得资源所有权

        raw_inputs[i]->erase(TASK_EVENT_KEY);
      } else {
        // 无资源所有权
        continue;
      }
    }

    return;
  };

#ifndef DOXYGEN_SHOULD_SKIP_THIS
  ~Batching();
#endif

#ifndef DOXYGEN_SHOULD_SKIP_THIS
 protected:
#endif

  virtual void run();

 private:
  uint32_t max_batch_size_{1};
  std::thread thread_;
  ThreadSafeSizedQueue<dict> input_queue_;
  ThreadSafeSizedQueue<std::vector<dict>> batched_queue_;
  float batching_timeout_;

  std::unique_ptr<Params> params_;
  std::string node_name_;
  std::unique_ptr<Backend> backend_;
  std::unique_ptr<Backend> cal_request_size_method_;

  std::atomic_bool bThreadInited_{false};

  std::exception_ptr init_eptr_;

  // std::atomic<unsigned> count_{0};

  std::shared_ptr<RuningState> runing_state_;
  int contiguous_batching_{0};

  std::unique_ptr<RequestStates> request_states_;
};
}  // namespace ipipe
