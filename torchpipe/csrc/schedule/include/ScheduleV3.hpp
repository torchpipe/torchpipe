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
#include <memory>
#include <string>
#include <vector>
#include "Backend.hpp"
#include "dict.hpp"
#include "event.hpp"
#include "params.hpp"
#include "reflect.h"
#include "threadsafe_queue.hpp"

#include "RuningState.hpp"

namespace ipipe {

class ScheduleV3 : public Backend {
 public:
  bool init(const std::unordered_map<std::string, std::string>& config, dict dict_config) override;

  virtual uint32_t max() const { return UINT32_MAX; };

  void forward(const std::vector<dict>& raw_inputs);

#ifndef DOXYGEN_SHOULD_SKIP_THIS
  ~ScheduleV3();
#endif

#ifndef DOXYGEN_SHOULD_SKIP_THIS
 protected:
#endif

  virtual void run();

 private:
  void async_forward(const std::vector<dict>& raw_inputs);

  uint32_t max_batch_size_{1};
  std::thread thread_;
  ThreadSafeQueue<dict> input_queue_;
  float batching_timeout_;

  std::unique_ptr<Params> params_;
  std::string node_name_;
  std::unique_ptr<Backend> backend_;

  std::atomic_bool bThreadInited_{false};

  std::shared_ptr<RuningState> runing_state_;
};
}  // namespace ipipe
