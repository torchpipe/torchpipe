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

#include "Jump.hpp"
#include "base_logging.hpp"
#include "reflect.h"
#include "event.hpp"
#include "ipipe_common.hpp"
#include "dict_helper.hpp"

namespace ipipe {

// 暂时先支持输入输出都为1个的子图跳转。
std::vector<dict> Jump::split(dict input) { return {input}; }

void Jump::merge(const std::vector<dict>& inputs, dict in_put) {
  IPIPE_ASSERT(inputs.size() == 1 && in_put == inputs[0]);
  return;
}

bool Jump::init(const std::unordered_map<std::string, std::string>& config, dict dict_config) {
  auto iter = dict_config->find("Interpreter");
  if (iter == dict_config->end()) {
    SPDLOG_ERROR("Interpreter not exists");
    return false;
  }
  interpreter_ = any_cast<Backend*>(iter->second);
  IPIPE_ASSERT(interpreter_);

  // std::string jump;
  auto iter_config = config.find("jump");
  if (iter_config != config.end()) {
    jump_ = any_cast<std::string>(iter_config->second);
  }
  iter_config = config.find("Jump::backend");
  if (iter_config != config.end()) {
    std::string jump = any_cast<std::string>(iter_config->second);
    if (!jump_.empty()) {
      IPIPE_ASSERT(jump_ == jump);
    } else {
      jump_ = jump;
    }
  }
  if (jump_.empty()) {
    SPDLOG_ERROR("parameter `jump` not set");
    return false;
  }

  return post_init(config, dict_config);
}

void Jump::forward(const std::vector<dict>& input_dicts) {
  for (const auto& input : input_dicts) {
    IPIPE_ASSERT(input->find(TASK_EVENT_KEY) == input->end());
    IPIPE_ASSERT(input->find(TASK_STACK_KEY) == input->end());
  }
  DictHelper guard(input_dicts);
  // guard.keep(TASK_DATA_KEY).keep("node_name");  //.lazy_erase(TASK_EVENT_KEY);
  guard.keep("node_name")
      .keep(TASK_REQUEST_SIZE_KEY)
      .erase(TASK_REQUEST_SIZE_KEY);  //.lazy_erase(TASK_EVENT_KEY);
                                      // todo merge TASK_REQUEST_SIZE_KEY to TASK_EVENT_KEY

  try {
    std::vector<std::vector<dict>> splits;
    std::vector<dict> total_splits;
    for (auto input : input_dicts) {
      auto split_data = split_wrapper(input);
      splits.push_back(split_data);
    }
    for (const auto& item : splits) {
      for (const auto& item_in : item) {
        total_splits.push_back(item_in);
      }
    }
    auto event = make_event(total_splits.size());
    DictHelper local_guard(total_splits);
    // local_guard.set(TASK_EVENT_KEY, event).set("node_name", jump_);
    local_guard.set("node_name", jump_);

    interpreter_->forward(total_splits);
    // event->Wait();

    for (std::size_t i = 0; i < splits.size(); ++i) {
      merge_wrapper(input_dicts[i], splits[i]);
    }
  } catch (...) {
    guard.erase(TASK_RESULT_KEY);
    std::rethrow_exception(std::current_exception());
  }
};

std::vector<dict> Jump::split_wrapper(dict input) {
  // const static std::set<std::string> keep_key({TASK_STACK_KEY, TASK_EVENT_KEY});
  // check stack
  IPIPE_ASSERT(input->find(TASK_EVENT_KEY) == input->end());
  IPIPE_ASSERT(input->find(TASK_STACK_KEY) == input->end());

  auto split_data = split(input);

  return split_data;
}

void Jump::merge_wrapper(dict input, const std::vector<dict>& split_data) {
  merge(split_data, input);
}

IPIPE_REGISTER(Backend, Jump, "Jump");

}  // namespace ipipe