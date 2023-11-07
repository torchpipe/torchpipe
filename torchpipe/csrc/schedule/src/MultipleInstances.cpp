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

#include "MultipleInstances.hpp"
#include <condition_variable>
#include <numeric>
#include <sstream>
#include <map>
#include "exception.hpp"
#include <mutex>

#include "base_logging.hpp"
#include "event.hpp"
#include "reflect.h"
#include "time_utils.hpp"
#include "InstanceHandler.hpp"
namespace ipipe {

class SharedBackendsMap {};
// instances_grp
// active_instances_grp
std::set<int> range_data(int num) {
  std::vector<int> vec(num);
  std::iota(vec.begin(), vec.end(), 0);
  std::set<int> total(vec.begin(), vec.end());
  return total;
}
bool MultipleInstances::init(const std::unordered_map<std::string, std::string>& config,
                             dict dict_config) {
  params_ = std::unique_ptr<Params>(new Params({{"MultipleInstances::backend", ""},
                                                {"instance_num", "1"},
                                                {"instances_grp", ""},
                                                {"active_instances_grp", "0"},
                                                {"borrow_from", ""}},
                                               {"node_name"}, {}, {}));
  if (!params_->init(config)) return false;

  std::string node_name = params_->at("node_name");

  TRACE_EXCEPTION(instance_num_ = std::stoi(params_->at("instance_num")));
  if (instance_num_ > 1024 || instance_num_ == 0) {
    SPDLOG_ERROR("instance_num wired: " + params_->at("instance_num"));
    return false;
  }

  int activate_grp_index{0};
  //
  TRACE_EXCEPTION(activate_grp_index = std::stoi(params_->at("active_instances_grp")));
  TRACE_EXCEPTION(instances_grp_ = str2set(params_->at("instances_grp"), ',', ';'));

  // TRACE_EXCEPTION(force_ranges_ = str2int(params_->at("force_range"), ',', ';'));

  // if (!force_ranges_.empty()) {
  //   auto num_grp = instances_grp_.empty() ? 1 : instances_grp_.size();
  //   IPIPE_ASSERT(force_ranges_.size() <= num_grp);
  //   while (force_ranges_.size() < num_grp) {
  //     force_ranges_.push_back(force_ranges_.back());
  //   }

  //   for (const auto& force_range : force_ranges_) {
  //     IPIPE_ASSERT(force_range.size() == 2);
  //     IPIPE_ASSERT(force_range[0] >= 1 && force_range[0] <= force_range[1]);
  //   }
  // }

  std::string borrow_from = params_->at("borrow_from");
  if (borrow_from.empty()) {
    auto total = range_data(instance_num_);

    if (instances_grp_.empty()) {
      instances_grp_.emplace_back(total);
    }

    IPIPE_ASSERT(activate_grp_index >= 0 && activate_grp_index < instances_grp_.size());

    for (const auto& instance : instances_grp_) {
      for (const auto& item : instance) {
        IPIPE_CHECK(total.count(item) != 0, "instances_grp include invalid instance index");
        total.erase(item);
      }
    }
    IPIPE_CHECK(total.empty(), "some elements not included in instances_grp");

    if (dict_config) {
      state_ = std::make_shared<StateEvents>(instances_grp_[activate_grp_index].size());
      (*dict_config)["_state_event"] = state_;
    }

    std::map<int, int> thread_index2instance_grp;

    all_backends_.resize(instance_num_);
    for (std::size_t i = 0; i < instances_grp_.size(); ++i) {
      grp_queues_.emplace_back(std::make_unique<ThreadSafeQueue<std::vector<dict>>>());
    }

    for (std::size_t i = 0; i < instances_grp_.size(); ++i) {
      std::vector<Backend*> si;

      for (const auto& item : instances_grp_[i]) {
        thread_index2instance_grp[item] = i;
        if (!params_->at("MultipleInstances::backend").empty()) {
          all_backends_[item].reset(
              IPIPE_CREATE(Backend, params_->at("MultipleInstances::backend")));
        } else {
          all_backends_[item] = (std::make_unique<InstanceHandler>());
        }
        si.emplace_back(all_backends_[item].get());
      }
      std::lock_guard<std::mutex> lock_tmp(lock_);
      shared_instances_[node_name + "." + std::to_string(i)] = si;
    }
    dict dict_config_split =
        dict_config ? std::make_shared<std::unordered_map<std::string, any>>(*dict_config)
                    : std::make_shared<std::unordered_map<std::string, any>>();
    for (std::size_t i = 0; i < instance_num_; ++i) {
      (*dict_config_split)["_batched_queue"] = grp_queues_[thread_index2instance_grp[i]].get();

      auto new_config = config;
      new_config["_independent_thread_index"] = std::to_string(i);
      if (!all_backends_[i] || !all_backends_[i]->init(new_config, dict_config_split)) {
        return false;
      }
    }
    {
      std::lock_guard<std::mutex> lock(lock_);
      active_backends_ = shared_instances_.at(node_name + "." + std::to_string(activate_grp_index));
      // batched_queue_ = grp_queues_[activate_grp_index].get();
    }
  } else {
    {
      std::lock_guard<std::mutex> alock(lock_);
      TRACE_EXCEPTION(active_backends_ = shared_instances_.at(borrow_from + "." +
                                                              std::to_string(activate_grp_index)));
    }
  }

  // todo: active_instances => instances_grps

  std::vector<uint32_t> mins;
  std::vector<uint32_t> maxs;
  for (std::size_t i = 0; i < active_backends_.size(); ++i) {
    mins.push_back(active_backends_[i]->min());
    maxs.push_back(active_backends_[i]->max());
  }
  min_ = *std::min_element(mins.begin(), mins.end());
  max_ = *std::max_element(maxs.begin(), maxs.end());

  if (min() > max()) {
    SPDLOG_ERROR("MultipleInstances: min() > max().  min() = {} max() = {}", min(), max());
    return false;
  }
  sorted_max_ = sort_indexes(active_backends_);
  return true;
}
std::unordered_map<std::string, std::vector<Backend*>> MultipleInstances::shared_instances_{};
std::mutex MultipleInstances::lock_;
}  // namespace ipipe