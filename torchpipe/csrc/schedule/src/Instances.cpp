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

#include "Instances.hpp"
#include <condition_variable>
#include <numeric>
#include <sstream>
#include "exception.hpp"

#include "base_logging.hpp"
#include "event.hpp"
#include "reflect.h"
#include "time_utils.hpp"
#include "InstanceHandler.hpp"
namespace ipipe {

bool Instances::init(const std::unordered_map<std::string, std::string>& config, dict dict_config) {
  params_ = std::unique_ptr<Params>(
      new Params({{"Instances::backend", ""}, {"instance_num", "1"}}, {}, {}, {}));
  if (!params_->init(config)) return false;

  TRACE_EXCEPTION(instance_num_ = std::stoi(params_->at("instance_num")));
  if (instance_num_ > 1024 || instance_num_ == 0) {
    SPDLOG_ERROR("instance_num wired: " + params_->at("instance_num"));
    return false;
  }
  if (dict_config) {
    state_ = std::make_shared<StateEvents>(instance_num_);
    (*dict_config)["_state_event"] = state_;
  }

  for (std::size_t i = 0; i < instance_num_; ++i) {
    if (!params_->at("Instances::backend").empty()) {
      backends_.emplace_back(IPIPE_CREATE(Backend, params_->at("Instances::backend")));
    } else {
      backends_.emplace_back(std::make_unique<InstanceHandler>());
    }
  }

  dict dict_config_split =
      dict_config ? std::make_shared<std::unordered_map<std::string, any>>(*dict_config)
                  : std::make_shared<std::unordered_map<std::string, any>>();
  (*dict_config_split)["_batched_queue"] = &batched_queue_;
  int i = 0;
  for (auto& back_end : backends_) {
    auto new_config = config;
    new_config["_independent_thread_index"] = std::to_string(i++);
    // new_config["instance_num"] = params_->at("instance_num");
    if (!back_end || !back_end->init(new_config, dict_config_split)) {
      return false;
    }
  }
  if (min() > max() || min() != 1) {
    SPDLOG_ERROR("Instances: min() > max() || min() != 1.  min() = {} max() = {}", min(), max());
    return false;
  }
  sorted_max_ = sort_indexes(backends_);
  return true;
}

}  // namespace ipipe