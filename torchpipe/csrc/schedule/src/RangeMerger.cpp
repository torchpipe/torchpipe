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

#include "RangeMerger.hpp"
#include <cassert>
#include "base_logging.hpp"
#include "dict_helper.hpp"
#include "params.hpp"
#include "reflect.h"
#include "time_utils.hpp"
#include "MultipleInstances.hpp"
#include "base_logging.hpp"
#include <numeric>
#include <algorithm>

namespace ipipe {

bool RangeMerger::init(const std::unordered_map<std::string, std::string>& config_param,
                       dict dict_config) {
  params_ = std::unique_ptr<Params>(new Params({{"RangeMerger::backend", ""}}, {}, {}, {}));
  if (!params_->init(config_param)) return false;

  std::unordered_map<std::string, std::vector<std::string>> configs;

  std::size_t max_split = 0;

  if (config_param.empty()) {
    SPDLOG_ERROR("empty config_param");
    return false;
  }

  for (auto iter = config_param.begin(); iter != config_param.end(); ++iter) {
    auto single_config = str_split(iter->second, '&');
    if (single_config.empty()) {
      single_config.push_back("");
      // return false;
    }

    configs[iter->first] = single_config;
    max_split = std::max(max_split, single_config.size());
  }

  for (auto& item : configs) {
    while (item.second.size() < max_split) {
      item.second.push_back(item.second.back());
    }
  }
  // while (engines.size() < max_split) {
  //   engines.push_back(engines.back());
  // }

  std::vector<std::unordered_map<std::string, std::string>> split_configs(max_split);
  for (std::size_t i = 0; i < max_split; ++i) {
    for (auto& item : configs) {
      split_configs[i][item.first] = item.second[i];
    }
    if (split_configs[i].find("RangeMerger::backend") == split_configs[i].end()) {
      split_configs[i]["RangeMerger::backend"] = "";
    }
  }

  for (std::size_t i = 0; i < max_split; ++i) {
    if (!split_configs[i]["RangeMerger::backend"].empty()) {
      backends_.emplace_back(IPIPE_CREATE(Backend, split_configs[i]["RangeMerger::backend"]));
    } else {
      backends_.emplace_back(std::make_unique<MultipleInstances>());
    }
  }

  for (std::size_t i = 0; i < backends_.size(); ++i) {
    dict dict_config_split =
        dict_config ? std::make_shared<std::unordered_map<std::string, any>>(*dict_config)
                    : std::make_shared<std::unordered_map<std::string, any>>();
    if (!backends_[i] || !backends_[i]->init(split_configs[i], dict_config_split)) {
      return false;
    }
  }

  std::vector<uint32_t> mins;
  for (std::size_t i = 0; i < backends_.size(); ++i) {
    mins.push_back(backends_[i]->min());
  }
  auto min_size = *std::min_element(mins.begin(), mins.end());
  std::vector<uint32_t> maxs;
  for (std::size_t i = 0; i < backends_.size(); ++i) {
    maxs.push_back(backends_[i]->max());
  }
  max_ = *std::max_element(maxs.begin(), maxs.end());

  if (min() > max() || min_size != 1) {
    SPDLOG_ERROR("RangeMerger: min() > max() || min() != 1.  min() = {} max() = {}", min(), max());
    return false;
  }
  sorted_max_ = sort_vector(backends_);
  SPDLOG_DEBUG("{}: [{} {}]", params_->at("RangeMerger::backend"), min(), max_);
  return true;
}

void RangeMerger::forward(const std::vector<dict>& input_dicts) {
  auto inputs = input_dicts;
  while (!inputs.empty()) {
    int index = 0;
    auto input_true = split_inputs(inputs, index);
    assert(!input_true.empty());
    backends_[index]->forward(input_true);  //

    //@todo: check input_queue and sorted_max_
  }
  return;
}

// IPIPE_REGISTER(Backend, RangeMerger, "RangeMerger");
}  // namespace ipipe