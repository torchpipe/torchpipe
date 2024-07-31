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

#include "ScheduleV3.hpp"

#include <chrono>
#include <functional>
#include <sstream>
#include <thread>
#include "base_logging.hpp"
#include "Backend.hpp"
#include "dict.hpp"
#include "Instances.hpp"
#include "time_utils.hpp"
#include "RangeMerger.hpp"
#include "base_logging.hpp"
#include "MultipleInstances.hpp"
#include "dict_helper.hpp"
namespace ipipe {

ScheduleV3::~ScheduleV3() {
  bThreadInited_.store(false);
  if (!input_queue_.empty()) {
    SPDLOG_ERROR("!input_queue_.empty()");
  }
  if (thread_.joinable()) {
    thread_.join();
  }
}

bool ScheduleV3::init(const std::unordered_map<std::string, std::string>& config,
                      dict dict_config) {
  params_ = std::unique_ptr<Params>(new Params(
      {{"Schedule::backend", ""}, {"batching_timeout", "0"}, {"node_name", ""}}, {}, {}, {}));
  if (config.empty()) {
    SPDLOG_ERROR("empty config. Only support single-node configuration.");
    return false;
  }
  if (!params_->init(config)) return false;
  auto batching_timeouts = str_split(params_->at("batching_timeout"), '&');
  batching_timeout_ = 0;
  if (batching_timeouts.size() > 1) {
    SPDLOG_WARN("batching_timeout has more than one value, only the biggest one will be used.");
  }
  for (const auto& item : batching_timeouts) {
    batching_timeout_ = std::max(batching_timeout_, std::stof(item));
  }
  node_name_ = params_->at("node_name");

  if (params_->at("Schedule::backend").empty()) {
    backend_ = std::make_unique<RangeMerger>();
  } else {
    backend_ = std::unique_ptr<Backend>(IPIPE_CREATE(Backend, params_->at("Schedule::backend")));
  }

  runing_state_ = std::make_shared<RuningState>();
  if (backend_ && backend_->init(config, dict_config)) {
    max_batch_size_ = backend_->max();
    if (max_batch_size_ == UINT32_MAX) {
      SPDLOG_INFO(node_name_ + ": max() == UINT32_MAX");
    }
    if (max_batch_size_ != 1) {
      bThreadInited_.store(true);
      thread_ = std::thread(&ScheduleV3::run, this);
    }
    SPDLOG_INFO("{}: max_batch_size={}", node_name_, max_batch_size_);

    return true;
  } else {
    return false;
  }
}

void ScheduleV3::run() {  // only one ScheduleV3 thread

  std::vector<dict> input_data;

  while (bThreadInited_.load()) {
    auto data_size = input_queue_.size();

    if (data_size + input_data.size() >= max_batch_size_) {
      while (input_data.size() < max_batch_size_) {
        input_data.push_back(input_queue_.WaitPop());
      }

      backend_->forward(input_data);
    } else if (data_size + input_data.size() == 0) {
      dict tmp_dict;
      if (!input_queue_.WaitForPop(tmp_dict,
                                   batching_timeout_)) {  // every batching_timeout_ ms check that
                                                          // whether bIbited_ is true.
        continue;
      }
      input_data.push_back(tmp_dict);
      continue;

    } else {
      // 保证input_data里有至少一个
      if (input_data.empty()) {
        input_data.push_back(input_queue_.WaitPop());
      }
      std::shared_ptr<SimpleEvents> event =
          any_cast<std::shared_ptr<SimpleEvents>>(input_data[0]->at(TASK_EVENT_KEY));
      auto time_es = event->time_passed();

      if (time_es < batching_timeout_ && !runing_state_->skip_waiting_for_batching()) {
        input_queue_.Wait(int(batching_timeout_ - time_es));
        continue;
      }

      while (!input_queue_.empty() && (input_data.size() < max_batch_size_)) {
        input_data.push_back(input_queue_.WaitPop());
      }
      backend_->forward(input_data);
    }
    input_data.clear();
  }  // end while
};

void ScheduleV3::async_forward(const std::vector<dict>& raw_inputs) {
  if (max_batch_size_ == 1) {
    for (auto raw_input : raw_inputs) {
      backend_->forward({raw_input});
    }
  } else {
    for (auto raw_input : raw_inputs) {
      input_queue_.Push(raw_input);  // todo 限制送入的不能超过最大值
    }
  }
}

void ScheduleV3::forward(const std::vector<dict>& raw_inputs) {
  DictHelper helper(raw_inputs);
  helper.erase(TASK_RESULT_KEY);

  std::vector<std::shared_ptr<SimpleEvents>> events;
  for (auto raw_input : raw_inputs) {
    auto iter = raw_input->find(TASK_EVENT_KEY);
    if (iter != raw_input->end()) {
      events.emplace_back(any_cast<std::shared_ptr<SimpleEvents>>(iter->second));
    }
  }

  std::shared_ptr<RuningStateMonitor> guard_state =
      std::make_shared<RuningStateMonitor>(runing_state_);

  if (events.size() == raw_inputs.size()) {
    events.back()->add_callback([guard_state]() { guard_state->del(); });

    // 注意：资源所有权问题， 从此刻起 对 raw_input 没有读写权限，
    // 除非event通知
    async_forward(raw_inputs);
    return;
  }
  IPIPE_ASSERT(events.empty(), "number of events must be 0 or inputs.size()");

  auto event = make_event(raw_inputs.size());
  event->add_callback([guard_state]() { guard_state->del(); });

  helper.set(TASK_EVENT_KEY, event).lazy_erase(TASK_EVENT_KEY);

  async_forward(raw_inputs);
  /*! WARNING  this thread no longer has access to raw_inputs, otherwise it will cause
   * multi-threaded access */
  event->Wait();

  return;
};
IPIPE_REGISTER(Backend, ScheduleV3, "ScheduleV3");

}  // namespace ipipe
