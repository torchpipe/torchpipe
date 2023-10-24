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

#include "SortSchedule.hpp"

#include <chrono>
#include <functional>
#include <sstream>
#include <thread>
#include "base_logging.hpp"
#include "Backend.hpp"
#include "dict.hpp"
#include "event.hpp"
#include "Instances.hpp"
#include "time_utils.hpp"
namespace ipipe {

bool SortSchedule::init(const std::unordered_map<std::string, std::string>& config,
                        dict dict_config) {
  params_ = std::unique_ptr<Params>(new Params({{"SortSchedule::backend", ""},
                                                {"batching_timeout", "0"},
                                                {"number_factor", "1.5"},
                                                {"node_name", ""}},
                                               {}, {}, {}));
  if (!params_->init(config)) return false;

  batching_timeout_ = std::stof(params_->at("batching_timeout"));
  number_factor_ = std::stof(params_->at("number_factor"));
  if (number_factor_ <= 0) {
    SPDLOG_ERROR("Illegal number_factor_: {}", number_factor_);
    return false;
  }
  node_name_ = params_->at("node_name");

  if (params_->at("SortSchedule::backend").empty()) {
    backend_ = std::make_unique<Instances>();
  } else {
    backend_ =
        std::unique_ptr<Backend>(IPIPE_CREATE(Backend, params_->at("SortSchedule::backend")));
  }
  if (!dict_config) dict_config = std::make_shared<std::unordered_map<std::string, any>>();

  if (backend_ && backend_->init(config, dict_config)) {
    max_batch_size_ = backend_->max();
    if (backend_->min() != 1) {
      SPDLOG_ERROR("i do not know how to handle single input. node_name : {}.", node_name_);
      backend_.reset();
      return false;
    }
    if (max_batch_size_ != 1) {
      bInited_.store(true);
      auto iter = dict_config->find("_state_event");
      if (iter != dict_config->end()) {
        state_ = any_cast<std::shared_ptr<StateEvents>>(iter->second);
      }
      if (!state_) {
        SPDLOG_ERROR("SortSchedule could found any valid StateEvents.");
        return false;
      }
      thread_ = std::thread(&SortSchedule::run, this);
    }

    return true;
  }
  return false;
};

void SortSchedule::forward(const std::vector<dict>& raw_inputs) {
  std::vector<std::shared_ptr<SimpleEvents>> events;
  for (auto raw_input : raw_inputs) {
    auto& map_data = *raw_input;
    map_data.erase(TASK_RESULT_KEY);

    auto iter = raw_input->find(TASK_EVENT_KEY);
    if (iter == raw_input->end()) {
      auto event = make_event();
      events.emplace_back(event);
      map_data[TASK_EVENT_KEY] = event;
    } else {
      events.emplace_back(nullptr);
    }
  }

  assert(events.size() == raw_inputs.size());

  if (backend_->max() == 1) {
    for (auto raw_input : raw_inputs) {
      backend_->forward({raw_input});  // 异步调用
    }
  } else {
    std::vector<ThreadSafeSortList<float, dict>::ScoreValue> new_value;
    for (auto raw_input : raw_inputs) {
      auto iter = raw_input->find("_sort_score");
      if (iter != raw_input->end()) {
        float score = any_cast<float>(iter->second);
        raw_input->erase(iter);
        // input_list_.Push(raw_input, score); // todo 限制送入的不能超过最大值
        new_value.emplace_back(ThreadSafeSortList<float, dict>::ScoreValue{raw_input, score});
      } else {
        float score = -1 * time_passed();
        // input_list_.Push(raw_input, score); // todo 限制送入的不能超过最大值
        new_value.emplace_back(ThreadSafeSortList<float, dict>::ScoreValue{raw_input, score});
      }
    }
    input_list_.Push(new_value);
  }

  for (std::size_t i = 0; i < raw_inputs.size(); ++i) {
    if (events[i]) {
      events[i]->Wait();
      // now you can handle raw_inputs again;
      raw_inputs[i]->erase(TASK_EVENT_KEY);
    }
  }

  return;
};

SortSchedule::~SortSchedule() {
  bInited_.store(false);
  if (!input_list_.empty()) {
    SPDLOG_ERROR("!input_list_.empty()");
  }
  if (thread_.joinable()) {
    thread_.join();
  }
}

void SortSchedule::run() {  // only one SortSchedule thread
  bool emergency = false;
  while (bInited_.load()) {
    StateEvents::State state = StateEvents::State::invalid;
    if (state_->WaitUnFull(100, state)) {  // todo instance_num==1

      if (!input_list_.WaitUnEmpty(100)) {
        emergency = true;
        continue;
      }
      std::vector<dict> input_data;
      if (state == StateEvents::State::empty && emergency)  // empty, 全部空， 不用计较超时时间
      {
        input_data = input_list_.WaitPopRightNow(100, max_batch_size_);
        if (!input_data.empty()) SPDLOG_DEBUG("WaitPopRightNow {}", input_data.size());

      } else if (state == StateEvents::State::empty ||
                 state == StateEvents::State::half) {  // 正常超时等待
        std::shared_ptr<SimpleEvents> event =
            any_cast<std::shared_ptr<SimpleEvents>>(input_list_.front()->at(TASK_EVENT_KEY));
        auto time_es = event->time_passed();
        input_data = input_list_.WaitTimeOut(  // WaitTimeOut WaitForPop
            std::max(0, int(batching_timeout_ - time_es)), max_batch_size_, number_factor_);
        SPDLOG_DEBUG("WaitForPop {}", input_data.size());
      } else {
        assert(false);
      }
      emergency = false;
      assert(input_data.size() <= max_batch_size_);
      if (input_data.empty()) continue;

      backend_->forward(input_data);
    }
  }  // end while
};
IPIPE_REGISTER(Backend, SortSchedule, "SortSchedule");

}  // namespace ipipe