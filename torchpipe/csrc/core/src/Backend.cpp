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

#include "Backend.hpp"
#include "event.hpp"
namespace ipipe {

class AsyncBackend : public Backend {
 public:
#ifndef DOXYGEN_SHOULD_SKIP_THIS
  virtual void forward(const std::vector<dict>& input_dicts) override final {
    for (auto input : input_dicts) {
      if (input->find(TASK_EVENT_KEY) != input->end())
        throw std::runtime_error("TASK_EVENT_KEY found");
    }
    auto event = make_event(input_dicts.size());
    for (auto input : input_dicts) {
      (*input)[TASK_EVENT_KEY] = event;
    }

    for (auto input : input_dicts) {
      forward(input);
    }
    auto exc = event->WaitAndGetExcept();

    for (auto input : input_dicts) {
      input->erase(TASK_EVENT_KEY);
    }

    if (exc) std::rethrow_exception(exc);
  };
#endif

  virtual void forward(dict input_dict) = 0;

  /// @return 1
  virtual uint32_t max() const override final { return UINT32_MAX; };
  /// @return 1
  virtual uint32_t min() const override final { return 1; };
};

}  // namespace ipipe