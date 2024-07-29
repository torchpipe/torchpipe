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

namespace ipipe {

// # cal_request_size_method = "AddRequestSizeTensor"

class Batching : public Backend {
 public:
  bool init(const std::unordered_map<std::string, std::string>& config, dict dict_config) {
    params_ = std::unique_ptr<Params>(
        new Params({{"multiple_instances", ""},
                    {"batching_timeout", "2"},
                    {"cal_request_size_method", ""},  // AddRequestSizeTensor
                    {"node_name", ""}},
                   {}, {}, {}));
    if (config.empty()) {
      SPDLOG_ERROR("empty config. Only support single-node configuration.");
      return false;
    }
    if (!params_->init(config)) return false;
    auto batching_timeouts = str_split(params_->at("batching_timeout"), '&');
    batching_timeout_ = 0;
    for (const auto& item : batching_timeouts) {
      batching_timeout_ = std::max(batching_timeout_, std::stof(item));
    }

    node_name_ = params_->at("node_name");

    if (params_->at("multiple_instances").empty()) {
      // backend_ = std::make_unique<RangeMerger>();
      backend_ = std::make_unique<MultiInstances>();
    } else {
      backend_ = std::unique_ptr<Backend>(IPIPE_CREATE(Backend, params_->at("multiple_instances")));
    }
    // batched_queue_ = std::make_unique<ThreadSafeSizedQueue<std::vector<dict>>>();
    (*dict_config)["_batched_queue"] = &batched_queue_;

    if (!params_->at("cal_request_size_method").empty()) {
      cal_request_size_method_ =
          std::unique_ptr<Backend>(IPIPE_CREATE(Backend, params_->at("cal_request_size_method")));
      IPIPE_ASSERT(cal_request_size_method_ && cal_request_size_method_->init(config, dict_config));
    }
    if (!backend_ || !backend_->init(config, dict_config)) return false;
    runing_state_ = std::make_shared<RuningState>();
    {
      max_batch_size_ = backend_->max();
      if (max_batch_size_ == UINT32_MAX) {
        SPDLOG_WARN(node_name_ + ": max() == UINT32_MAX");
      }

      if (max_batch_size_ != 1 && batching_timeout_ > 0) {
        bThreadInited_.store(true);
        thread_ = std::thread(&Batching::run, this);
      } else if (max_batch_size_ != 1 && batching_timeout_ == 0) {
        SPDLOG_WARN(
            "{}: Batching will not be enabled as batching_timeout is set to 0. Even though "
            "max_batch_size is greater than 1, multiple requests coming in simultaneously will not "
            "be batched together.",
            node_name_);
      }
      SPDLOG_INFO("{}: max_batch_size={}, batching_timeout={}", node_name_, max_batch_size_,
                  batching_timeout_);
    }
    return true;
  }

  /**
   * @return UINT32_MAX.
   */
  virtual uint32_t max() const { return UINT32_MAX; };

  void forward(const std::vector<dict>& raw_inputs) {
    if (cal_request_size_method_) {
      if (!bThreadInited_.load()) {
        SPDLOG_ERROR("cal_request_size_method_ is not supported when no batching needed");
        abort();
      }
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
        event->add_callback([guard_state]() { guard_state->del(); });
        if (cal_request_size_method_) {
          auto* data = raw_input.get();
          event->add_callback([data]() { data->erase(TASK_REQUEST_SIZE_KEY); });
        }
        map_data[TASK_EVENT_KEY] = event;
      } else {
        events.emplace_back(nullptr);

        std::shared_ptr<SimpleEvents> ev = any_cast<std::shared_ptr<SimpleEvents>>(iter->second);
        ev->add_callback([guard_state]() { guard_state->del(); });
        if (cal_request_size_method_) {
          auto* data = raw_input.get();
          ev->add_callback([data]() { data->erase(TASK_REQUEST_SIZE_KEY); });
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
};
}  // namespace ipipe
