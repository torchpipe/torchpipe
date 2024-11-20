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
#include "StatefulInstanceHandler.hpp"

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
        IPIPE_ASSERT(total.count(item) != 0, "instances_grp include invalid instance index");
        total.erase(item);
      }
    }
    IPIPE_ASSERT(total.empty(), "some elements not included in instances_grp");

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

void FakeInstances::forward(const std::vector<dict>& inputs_data) {
  const auto size = get_request_size(inputs_data);

  const auto index = get_best_match(size);
  SPDLOG_DEBUG("FakeInstances: size={} index={}", size, index);
  // auto inputs = inputs_data;
  if (size != inputs_data.size()) {
    IPIPE_ASSERT(index >= 0);  // garentied
    backends_[index]->forward(inputs_data);
  } else {
    if (index < 0) {
      backends_[0]->forward(inputs_data);
    }  // todo: split?
    else
      backends_[index]->forward(inputs_data);
  }

  // while (!inputs.empty()) {
  //   int index = 0;
  //   auto input_true = split_inputs(inputs, index);
  //   assert(!input_true.empty());
  // }
}

bool FakeInstances::init(const std::unordered_map<std::string, std::string>& config,
                         dict dict_config) {
  params_ = std::unique_ptr<Params>(
      new Params({{"instance_num", "1"}},
                 {"node_name", "FakeInstances::backend", "fake_instance_num"}, {}, {}));
  if (!params_->init(config)) return false;

  std::string node_name = params_->at("node_name");
  int instance_num{0};
  TRACE_EXCEPTION(instance_num = std::stoi(params_->at("instance_num")));
  IPIPE_ASSERT(instance_num == 1);
  TRACE_EXCEPTION(fake_instance_num_ = std::stoi(params_->at("fake_instance_num")));
  if (fake_instance_num_ > 1024 || fake_instance_num_ == 0) {
    SPDLOG_ERROR("fake_instance_num wired: " + params_->at("fake_instance_num"));
    return false;
  }

  // batched_queue_ =
  //     any_cast<ThreadSafeSizedQueue<std::vector<dict>>*>(dict_config->at("_batched_queue"));

  // all_backends_.resize(fake_instance_num_);
  for (std::size_t i = 0; i < fake_instance_num_; ++i) {
    backends_.emplace_back(IPIPE_CREATE(Backend, params_->at("FakeInstances::backend")));
  }  // todo RegisterInstances

  dict dict_config_split =
      dict_config ? std::make_shared<std::unordered_map<std::string, any>>(*dict_config)
                  : std::make_shared<std::unordered_map<std::string, any>>();
  for (std::size_t i = 0; i < fake_instance_num_; ++i) {
    auto new_config = config;
    new_config["_independent_thread_index"] = std::to_string(i);
    new_config["instance_num"] = std::to_string(fake_instance_num_);
    if (!backends_[i] || !backends_[i]->init(new_config, dict_config_split)) {
      SPDLOG_ERROR("FakeInstances: init failed");
      return false;
    }
  }

  std::vector<uint32_t> mins;
  std::vector<uint32_t> maxs;
  for (std::size_t i = 0; i < backends_.size(); ++i) {
    mins.push_back(backends_[i]->min());
    maxs.push_back(backends_[i]->max());
  }
  min_ = *std::min_element(mins.begin(), mins.end());
  max_ = *std::max_element(maxs.begin(), maxs.end());

  if (min() > max()) {
    SPDLOG_ERROR("FakeInstances: min() > max().  min() = {} max() = {}", min(), max());
    return false;
  }
  sorted_max_ = sort_indexes(backends_);
  return true;
}

bool MultiInstances::init(const std::unordered_map<std::string, std::string>& config,
                          dict dict_config) {
  params_ = std::unique_ptr<Params>(
      new Params({{"instance_num", "1"}, {"instance_handle", ""}}, {"node_name"}, {}, {}));
  if (!params_->init(config)) return false;

  std::string node_name = params_->at("node_name");

  TRACE_EXCEPTION(instance_num_ = std::stoi(params_->at("instance_num")));
  if (instance_num_ > 1024 || instance_num_ == 0) {
    SPDLOG_ERROR("instance_num wired: " + params_->at("instance_num"));
    return false;
  }
  IPIPE_ASSERT(dict_config);
  // if (dict_config) {
  //   state_ = std::make_shared<StateEvents>(instance_num_);
  //   (*dict_config)["_state_event"] = state_;
  // }

  all_backends_.resize(instance_num_);
  // batched_queue_ = std::make_unique<ThreadSafeSizedQueue<std::vector<dict>>>();
  batched_queue_ =
      any_cast<ThreadSafeSizedQueue<std::vector<dict>>*>(dict_config->at("_batched_queue"));

  for (auto& backend : all_backends_) {
    if (!params_->at("instance_handle").empty()) {
      backend = std::unique_ptr<Backend>(IPIPE_CREATE(Backend, params_->at("instance_handle")));
    } else {
      backend = std::make_unique<StatefulInstanceHandler>();
    }
  }

  dict dict_config_split =
      dict_config ? std::make_shared<std::unordered_map<std::string, any>>(*dict_config)
                  : std::make_shared<std::unordered_map<std::string, any>>();

  for (std::size_t i = 0; i < instance_num_; ++i) {
    // (*dict_config_split)["_batched_queue"] = batched_queue_.get();

    auto new_config = config;
    new_config["_independent_thread_index"] = std::to_string(i);
    if (!all_backends_[i] || !all_backends_[i]->init(new_config, dict_config_split)) {
      return false;
    }
  }

  std::vector<uint32_t> mins;
  std::vector<uint32_t> maxs;
  for (std::size_t i = 0; i < all_backends_.size(); ++i) {
    mins.push_back(all_backends_[i]->min());
    maxs.push_back(all_backends_[i]->max());
  }
  min_ = *std::min_element(mins.begin(), mins.end());
  max_ = *std::max_element(maxs.begin(), maxs.end());

  if (min() > max()) {
    SPDLOG_ERROR("MultipleInstances: min() > max().  min() = {} max() = {}", min(), max());
    return false;
  }
  return true;
}

void MultiInstances::forward(const std::vector<dict>& inputs_data) {
  const auto size = get_request_size(inputs_data);
  IPIPE_ASSERT(size > 0 && inputs_data.size() > 0);
  // SPDLOG_INFO("MultiInstances: request size={}", size);
  batched_queue_->Push(inputs_data, size);
}

IPIPE_REGISTER(Backend, FakeInstances, "FakeInstances");
}  // namespace ipipe