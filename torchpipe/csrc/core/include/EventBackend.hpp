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
#include "Backend.hpp"
#include "event.hpp"
namespace ipipe {
class SingleEventBackend : public SingleBackend {
 public:
  /**
   * 子类重新实现了此函数。
   */
  virtual void forward(dict input_dict, std::shared_ptr<SimpleEvents> event,
                       std::string node_name) noexcept = 0;
  virtual void forward(dict input) override {
    assert(input != nullptr);
    input->erase(TASK_RESULT_KEY);
    std::string* node_name{nullptr};

    auto iter = input->find("node_name");
    if (iter != input->end()) {
      node_name = any_cast<std::string>(&iter->second);
    }

    iter = input->find(TASK_EVENT_KEY);
    if (iter == input->end()) {
      if (!node_name) {
        throw std::runtime_error("there is no `node_name`");
      }
      iter = input->find(TASK_DATA_KEY);
      if (iter == input->end()) {
        throw std::runtime_error("the input data should contain TASK_DATA_KEY");
      }
      auto event = make_event();
      (*input)[TASK_EVENT_KEY] = event;

      forward(input, event, *node_name);
      input->erase(TASK_EVENT_KEY);
      event->Wait();
      return;

    } else {
      // if event exists, it should hold all exceptions.
      std::shared_ptr<SimpleEvents> event = any_cast<std::shared_ptr<SimpleEvents>>(iter->second);
      assert(event != nullptr);
      if (!node_name)
        event->set_exception_and_notify_all(
            std::make_exception_ptr(std::runtime_error("there is no `node_name`")));

      iter = input->find(TASK_DATA_KEY);
      if (iter == input->end()) {
        event->set_exception_and_notify_all(std::make_exception_ptr(
            std::runtime_error("the input data should contain TASK_DATA_KEY")));
        return;
      }

      forward(input, event, *node_name);
    }
  }
};
}  // namespace ipipe