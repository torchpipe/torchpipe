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

#include "Schedule.hpp"

#include <chrono>
#include <functional>
#include <sstream>
#include <thread>
#include "base_logging.hpp"
#include "Backend.hpp"
#include "dict.hpp"
#include "Instances.hpp"
#include "time_utils.hpp"
#include "reflect.h"
#include "dict_helper.hpp"

namespace ipipe {

Schedule::~Schedule() {
  while (!input_queue_.empty()) {
    SPDLOG_WARN("!input_queue_.empty(). waiting...");
    SPDLOG_ERROR("You must wait for all tasks to complete before deleting the system.");
    std::this_thread::yield();
  }
  bInited_.store(false);

  if (thread_.joinable()) {
    thread_.join();
  }
}

bool Schedule::init(const std::unordered_map<std::string, std::string>& config, dict dict_config) {
  params_ = std::unique_ptr<Params>(new Params(
      {{"Schedule::backend", ""}, {"batching_timeout", "0"}, {"node_name", ""}}, {}, {}, {}));
  if (!params_->init(config)) return false;

  batching_timeout_ = std::stof(params_->at("batching_timeout"));
  node_name_ = params_->at("node_name");

  if (params_->at("Schedule::backend").empty()) {
    backend_ = std::make_unique<Instances>();
  } else {
    backend_ = std::unique_ptr<Backend>(IPIPE_CREATE(Backend, params_->at("Schedule::backend")));
  }
  if (backend_ && backend_->init(config, dict_config)) {
    max_batch_size_ = backend_->max();
    if (max_batch_size_ == UINT32_MAX) {
      SPDLOG_WARN(node_name_ + ": max() == UINT32_MAX");
    }
    if (max_batch_size_ != 1) {
      bInited_.store(true);
      thread_ = std::thread(&Schedule::run, this);
    }

    return true;
  } else {
    return false;
  }
}

void Schedule::run() {  // there is only one Schedule thread for one node

  std::vector<dict> input_data;

  while (bInited_.load()) {
    auto data_size = input_queue_.size();

    if (data_size + input_data.size() >= max_batch_size_) {
      // for (uint32_t i = 0; i < max_batch_size_ - input_data.size(); ++i) {
      //   input_data.push_back(input_queue_.WaitPop());
      // }
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
        assert(!input_queue_.empty());
        input_data.push_back(input_queue_.WaitPop());
      }
      std::shared_ptr<SimpleEvents> event =
          any_cast<std::shared_ptr<SimpleEvents>>(input_data[0]->at(TASK_EVENT_KEY));
      auto time_es = event->time_passed();

      if (time_es < batching_timeout_) {
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

void Schedule::async_forward(const std::vector<dict>& raw_inputs) {
  if (backend_->max() == 1) {
    for (auto raw_input : raw_inputs) {
      backend_->forward({raw_input});
    }
  } else {
    for (auto raw_input : raw_inputs) {
      input_queue_.Push(raw_input);  // todo 限制送入的不能超过最大值
    }
  }
}

void Schedule::forward(const std::vector<dict>& raw_inputs) {
  uint32_t num_events = 0;
  for (auto raw_input : raw_inputs) {
    auto& map_data = *raw_input;
    map_data.erase(TASK_RESULT_KEY);

    auto iter = raw_input->find(TASK_EVENT_KEY);
    if (iter != raw_input->end()) {
      num_events++;
    }
  }
  if (num_events == raw_inputs.size()) {
    // 注意：资源所有权问题， 从此刻起 对 raw_input 没有读写权限，
    // 除非event通知
    async_forward(raw_inputs);
    return;
  }
  IPIPE_CHECK(num_events == 0, "num_events must be 0 or inputs.size()");

  auto event = make_event(raw_inputs.size());

  DictHelper d(raw_inputs);
  d.set(TASK_EVENT_KEY, event).lazy_erase(TASK_EVENT_KEY);

  async_forward(raw_inputs);
  /*! WARNING  this thread no longer has access to raw_inputs, otherwise it will cause
   * multi-threaded access */
  event->Wait();

  return;
};

IPIPE_REGISTER(Backend, Schedule, "Schedule");

}  // namespace ipipe