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

#include "Sequential.hpp"
#include <cassert>
#include "base_logging.hpp"
#include "dict_helper.hpp"
#include "params.hpp"
#include "reflect.h"
#include "time_utils.hpp"

namespace ipipe {

namespace {

std::unordered_map<std::string, std::vector<std::string>> split_config(
    const std::unordered_map<std::string, std::string>& config, char split_char,
    std::size_t target_len) {
  std::unordered_map<std::string, std::vector<std::string>> results;
  for (const auto& item : config) {
    results[item.first] = str_split(item.second, split_char, true);
    if (target_len != results[item.first].size() && 1 != results[item.first].size()) {
      std::stringstream ss;
      ss << "split `" << item.second << "` by `" << split_char << "`, and we get ";
      ss << results[item.first].size() << " configuration, but 1 or " << target_len << " needed.";
      throw std::runtime_error(ss.str());
    }
    std::reverse(results[item.first].begin(), results[item.first].end());
  }
  return results;
}
std::unordered_map<std::string, std::string> get_config(
    std::unordered_map<std::string, std::vector<std::string>>& configs, std::size_t i) {
  std::unordered_map<std::string, std::string> result;
  for (auto& item : configs) {
    if (item.second.size() == 1)
      result[item.first] = item.second.back();
    else if (item.second.size() > i) {
      result[item.first] = item.second[i];
    } else if (item.second.empty()) {
      result[item.first] = "";
    } else {
      throw std::runtime_error(item.first + ": not enough parameter");
    }
  }
  return result;
}
}  // namespace

bool Sequential::init(const std::unordered_map<std::string, std::string>& config_param,
                      dict dict_config) {
  register_name_ = IPIPE_GET_REGISTER_NAME(Backend, this);
  if (register_name_.empty()) register_name_ = "Sequential";
  params_ = std::unique_ptr<Params>(new Params({}, {register_name_ + "::backend"}, {}, {}));
  if (!params_->init(config_param)) return false;
  engine_names_ =
      str_split_brackets_match(params_->at(register_name_ + "::backend"), ',', '[', ']');
  if (engine_names_.empty()) {
    SPDLOG_ERROR(register_name_ + "::backend: not set.");
    return false;
  }
  // 倒序执行初始化
  std::reverse(engine_names_.begin(), engine_names_.end());
  std::size_t num_one = 0;
  std::size_t num_special_max = 0;
  auto configs = split_config(config_param, '|', engine_names_.size());
  configs.erase(register_name_ + "::backend");

  for (std::size_t i = 0; i < engine_names_.size(); ++i) {
    const auto& engine = engine_names_[i];
    std::string pre_str;
    auto backend = pre_parentheses_split(engine, pre_str);
    // auto new_config = config_param;
    auto new_config = get_config(configs, i);

    brackets_split(backend, new_config);
    engines_.emplace_back(IPIPE_CREATE(Backend, new_config.at("backend")));

    if (pre_str.empty()) {
      if (engines_.size() == engine_names_.size()) {
        pre_str = "Run";
      } else {
        pre_str = "swap";
      }
    }
    filters_.emplace_back(IPIPE_CREATE(Filter, pre_str));

    // try {
    if (!engines_.back() || !engines_.back()->init(new_config, dict_config)) {
      engines_.clear();
      return false;
    }
    if (!filters_.back() || !filters_.back()->init(new_config, dict_config)) {
      filters_.clear();
      return false;
    }
    // } catch (const std::exception& e) {
    //   SPDLOG_ERROR(engine + "(filter={}): init failed. " + e.what(), pre_str);
    //   engines_.clear();
    //   filters_.clear();
    //   return false;
    // }
    uint32_t local_max = engines_.back()->max();
    uint32_t local_min = engines_.back()->min();
    SPDLOG_DEBUG("{}->max()= {}", new_config.at("backend"), local_max);
    // if ((local_max == 1)) {
    //   assert(engines_.back()->min() == 1);
    //   local_max = UINT32_MAX;
    // }
    if (local_max == 1) {
      num_one++;
      IPIPE_ASSERT(local_min == 1);
    } else {
      if (local_max != UINT32_MAX) {
        num_special_max++;
      }
      min_ = std::max(min_, local_min);
      max_ = std::min(max_, local_max);
    }
  }
  if (num_one == engine_names_.size()) {
    min_ = 1;
    max_ = 1;
  }
  // if (max_ == UINT32_MAX) {
  //   max_ = 1;
  // }

  if (num_special_max > 1) {
    SPDLOG_ERROR(
        "Sequential[{}]: it is not allowed to have multiple nontrivial("
        "`max() != 1 && max() != UINT32_MAX`) backends.",
        params_->at(register_name_ + "::backend"));
    return false;
  }

  if (min_ > max_) {
    SPDLOG_ERROR(" min() > max() : min()= {} max() = {}. backend={}", min_, max_,
                 params_->at(register_name_ + "::backend"));
    return false;
  }

  // if (min_ != 1) {
  //   SPDLOG_ERROR(" min() must be 1: min()= {}. backend={}", min_,
  //                params_->at(register_name_ + "::backend"));
  //   return false;
  // }
  // forward需要顺序执行
  std::reverse(engines_.begin(), engines_.end());
  std::reverse(filters_.begin(), filters_.end());
  std::reverse(engine_names_.begin(), engine_names_.end());
  // engine_names_.push_back("NULL");

  SPDLOG_INFO("{}: [{} {}]", params_->at(register_name_ + "::backend"), min_, max_);

  return true;
}

void Sequential::forward(const std::vector<dict>& input_dicts) {
  DictHelper dicts_guard(input_dicts);
  dicts_guard.keep(TASK_DATA_KEY);  // to keey the storage of TASK_DATA_KEY. This tensor is created
                                    // in another stream
  std::set<std::size_t> break_index;
  for (std::size_t i = 0; i < engines_.size(); ++i) {
    // filters
    std::vector<dict> valid_inputs;
    for (std::size_t j = 0; j < input_dicts.size() && break_index.count(j) == 0; ++j) {
      const auto input_dict = input_dicts[j];
      auto filter_result = filters_[i]->forward(input_dict);
      if (filter_result == Filter::status::Run) {
        valid_inputs.push_back(input_dict);
        continue;
      } else if (filter_result == Filter::status::Skip) {
        continue;
      } else if (filter_result == Filter::status::Break ||
                 filter_result == Filter::status::SerialSkip ||
                 filter_result == Filter::status::SubGraphSkip) {
        break_index.insert(j);
        if (break_index.size() == input_dicts.size()) return;
        continue;

      } else if (filter_result == Filter::status::Error) {
        dicts_guard.erase(TASK_RESULT_KEY);
        std::string msg = "forward stoped: previous:";
        if (i != 0) {
          msg += engine_names_[i - 1] + ", now:" + engine_names_[i];
        } else {
          msg += ": " + engine_names_[i];
        }
        throw std::runtime_error(msg);
      } else {
        throw std::runtime_error("unsupported filter result -> " +
                                 std::to_string(static_cast<int>(filter_result)));
      }
    }
    if (valid_inputs.empty()) continue;

    auto& engine = engines_[i];

    if (valid_inputs.size() > engine->max()) {  // todo 支持[1,x]
                                                // 类型的输入分批动态组batch
      assert(1 == engine->max());
      for (auto input : valid_inputs) {
        engine->forward({input});
      }
    } else {
      engine->forward(valid_inputs);
    }
  }

  return;
}

IPIPE_REGISTER(Backend, Sequential, "Sequential,S");
// IPIPE_REGISTER(Backend, Sequential, "S");

}  // namespace ipipe