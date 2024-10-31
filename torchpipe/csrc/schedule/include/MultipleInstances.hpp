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

#include <atomic>
#include <cassert>
#include <memory>
#include <numeric>
#include <string>
#include <unordered_map>

#include <vector>
#include "time_utils.hpp"

#include <algorithm>
#include "Backend.hpp"
#include "dict.hpp"
#include "params.hpp"
#include "reflect.h"
#include "state_event.hpp"
#include "threadsafe_queue.hpp"
#include "threadsafe_queue_sized.hpp"

namespace ipipe {

/**
 * @brief (WIP)提供多实例功能，与 Schedule
 * SortSchedule 等配合使用，形成完整的单节点调度功能。
 */
class MultipleInstances : public Backend {
 public:
  /**
   * @brief
   *
   * @param MultipleInstances::backend
   * 实现backend消费逻辑的子后端。默认为 InstanceHandler.
   *
   *
   */
  bool init(const std::unordered_map<std::string, std::string>& config, dict dict_config) override;

  /**
   * @brief
   * 将输入数据按照子后端输入范围分隔到队列中。实例可能输入范围不同，尝试优先分配给max()大的实例。
   *
   */
  void forward(const std::vector<dict>& inputs_data) {
    auto inputs = inputs_data;
    while (!inputs.empty()) {
      int index = 0;
      auto input_true = split_inputs(inputs, index);
      assert(!input_true.empty());

      active_backends_[index]->forward(input_true);
      // batched_queue_.PushIfEmpty(input_true);
    }
  }

  /// 子Backend的上限最大值
  uint32_t max() const override { return max_; }

  /// 子Backend的下限最小值
  uint32_t min() const override { return min_; }

#ifndef DOXYGEN_SHOULD_SKIP_THIS
  // 先销毁消费线程， 确保销毁顺序
  ~MultipleInstances() { all_backends_.clear(); }
#endif
 private:
  // from
  // https://stackoverflow.com/questions/1577475/c-sorting-and-keeping-track-of-indexes
  template <typename T>
  std::vector<std::size_t> sort_indexes(const std::vector<T>& v) {
    std::vector<size_t> idx(v.size());
    std::iota(idx.begin(), idx.end(), 0);

    std::stable_sort(idx.begin(), idx.end(),
                     [&v](size_t i1, size_t i2) { return v[i1]->max() > v[i2]->max(); });

    return idx;
  }
  int get_best_match(std::size_t size_of_input) {
    for (auto item : sorted_max_) {
      if (size_of_input >= active_backends_[item]->min()) {
        return item;
      }
    }
    return -1;
  }

  std::vector<dict> split_inputs(std::vector<dict>& inputs, int& index) {
    assert(!inputs.empty());
    index = get_best_match(inputs.size());
    assert(index >= 0);
    assert(inputs.size() >= active_backends_[index]->min());

    if (inputs.size() <= active_backends_[index]->max()) {
      std::vector<dict> out;
      std::swap(out, inputs);
      return out;
    } else {
      std::vector<dict> out(inputs.begin(), inputs.begin() + active_backends_[index]->max());
      inputs = std::vector<dict>(inputs.begin() + active_backends_[index]->max(), inputs.end());
      return out;
    }
  }

 private:
  std::unique_ptr<Params> params_;
  std::vector<Backend*> active_backends_;
  std::vector<std::unique_ptr<Backend>> all_backends_;
  // ThreadSafeQueue<std::vector<dict>> batched_queue_;
  std::vector<std::unique_ptr<ThreadSafeQueue<std::vector<dict>>>> grp_queues_;
  // ThreadSafeQueue<std::vector<dict>>* batched_queue_{nullptr};
  uint32_t instance_num_;
  std::shared_ptr<StateEvents> state_;
  uint32_t max_{0};
  uint32_t min_{0};

  std::vector<std::size_t> sorted_max_;

  std::vector<std::set<int>> instances_grp_;

  std::unordered_map<std::string, std::string> config_;

  // class SharedInstances {
  //  public:
  //   ThreadSafeQueue<std::vector<dict>>* queue;
  //   std::vector<Backend*> backends;
  // };
  static std::mutex lock_;
  static std::unordered_map<std::string, std::vector<Backend*>> shared_instances_;
};

/**
 * @brief (WIP)提供多实例功能，与 Schedule
 * SortSchedule 等配合使用，形成完整的单节点调度功能。
 */
class FakeInstances : public Backend {
 public:
  bool init(const std::unordered_map<std::string, std::string>& config, dict dict_config) override;

  void forward(const std::vector<dict>& inputs_data);

  /// 子Backend的上限最大值
  uint32_t max() const override { return max_; }

  /// 子Backend的下限最小值
  uint32_t min() const override { return min_; }

#ifndef DOXYGEN_SHOULD_SKIP_THIS
  // 先销毁消费线程， 确保销毁顺序
  ~FakeInstances() { backends_.clear(); }
#endif
 private:
  // from
  // https://stackoverflow.com/questions/1577475/c-sorting-and-keeping-track-of-indexes
  template <typename T>
  std::vector<std::size_t> sort_indexes(const std::vector<T>& v) {
    std::vector<size_t> idx(v.size());
    std::iota(idx.begin(), idx.end(), 0);

    std::stable_sort(idx.begin(), idx.end(),
                     [&v](size_t i1, size_t i2) { return v[i1]->max() < v[i2]->max(); });

    return idx;
  }
  int get_best_match(std::size_t size_of_input) {
    for (auto item : sorted_max_) {
      if (size_of_input >= backends_[item]->min() && (size_of_input <= backends_[item]->max())) {
        return item;
      }
    }
    return -1;
  }

 private:
  std::unique_ptr<Params> params_;
  std::vector<std::unique_ptr<Backend>> backends_;

  uint32_t fake_instance_num_;
  uint32_t max_{0};
  uint32_t min_{0};

  std::vector<std::size_t> sorted_max_;

  // std::unordered_map<std::string, std::string> config_;
};

class MultiInstances : public Backend {
 public:
  bool init(const std::unordered_map<std::string, std::string>& config, dict dict_config) override;

  void forward(const std::vector<dict>& inputs_data) override;

  uint32_t max() const override { return max_; }

  /// 子Backend的下限最小值
  uint32_t min() const override { return min_; }
  // important: destroy all_backends_ befire batched_queue_
  ~MultiInstances() { all_backends_.clear(); }

 private:
  std::unique_ptr<Params> params_;
  // std::vector<Backend*> active_backends_;
  std::vector<std::unique_ptr<Backend>> all_backends_;
  // std::vector<std::unique_ptr<Backend>> all_backends_;
  // std::unique_ptr<Backend> backend_;
  ThreadSafeSizedQueue<std::vector<dict>>* batched_queue_ = nullptr;
  // std::unique_ptr<ThreadSafeSizedQueue<std::vector<dict>>> batched_queue_;
  // ThreadSafeQueue<std::vector<dict>>* batched_queue_{nullptr};
  uint32_t instance_num_;
  std::shared_ptr<StateEvents> state_;
  uint32_t max_{0};
  uint32_t min_{0};

  // std::unordered_map<std::string, std::string> config_;
};
}  // namespace ipipe