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

#include "JumpV0.hpp"
#include "base_logging.hpp"
#include "reflect.h"
#include "event.hpp"
namespace ipipe {

bool JumpV0::init(const std::unordered_map<std::string, std::string>& config, dict dict_config) {
  auto iter = config.find("JumpV0::backend");
  if (iter == config.end()) {
    SPDLOG_ERROR("JumpV0::backend not exists");
    return false;
  }
  node_name_ = iter->second;

  auto iter_dict = dict_config->find("Interpreter");
  if (iter_dict == dict_config->end()) {
    SPDLOG_ERROR("Interpreter not exists");
    return false;
  }
  pInterpreter = any_cast<Backend*>(iter_dict->second);
  // backend_ = std::unique_ptr<Backend>(IPIPE_CREATE(Backend, iter->second));
  // if (backend_ && backend_->init())
  //   return true;
  return pInterpreter;
}

void JumpV0::forward(const std::vector<dict>& inputs) {
  std::vector<std::string> node_names;
  std::vector<any> pstacks;

  for (auto item : inputs) {
    auto iter = item->find("node_name");
    if (iter != item->end()) {
      std::string node_name = any_cast<std::string>(iter->second);
      node_names.push_back(node_name);
    } else {
      node_names.emplace_back("");
    }
    (*item)["node_name"] = node_name_;
  }

  for (auto item : inputs) {
    auto iter_stack = item->find(TASK_STACK_KEY);
    if (iter_stack != item->end()) {
      pstacks.emplace_back(iter_stack->second);
      item->erase(iter_stack);
    } else {
      pstacks.emplace_back(any());
    }
  }
  auto inner_event = make_event(inputs.size());

  std::vector<std::shared_ptr<SimpleEvents>> events;
  for (auto raw_input : inputs) {
    auto& map_data = *raw_input;

    auto iter = raw_input->find(TASK_EVENT_KEY);
    if (iter != raw_input->end()) {
      assert(false);
      std::shared_ptr<SimpleEvents> event = any_cast<std::shared_ptr<SimpleEvents>>(iter->second);
      events.emplace_back(event);
      map_data.erase(iter);

    } else {
      events.emplace_back(nullptr);
    }
    map_data[TASK_EVENT_KEY] = inner_event;
  }

  pInterpreter->forward(inputs);

  inner_event->WaitAndGetExcept();  // not rethrow_exception here

  for (std::size_t i = 0; i < pstacks.size(); ++i) {
    if (pstacks[i].has_value()) {
      (*inputs[i])[TASK_STACK_KEY] = pstacks[i];
    } else {
      assert(false);
    }
  }

  for (std::size_t i = 0; i < events.size(); ++i) {
    auto& map_data = *inputs[i];
    if (events[i]) {
      map_data[TASK_EVENT_KEY] = events[i];
    } else {
      map_data.erase(TASK_EVENT_KEY);
    }

    if (!node_names[i].empty()) {
      map_data["node_name"] = node_names[i];
    }
  }

  if (inner_event->has_exception()) {
    std::rethrow_exception(inner_event->reset_exception());
  }
}

IPIPE_REGISTER(Backend, JumpV0, "JumpV0");

}  // namespace ipipe