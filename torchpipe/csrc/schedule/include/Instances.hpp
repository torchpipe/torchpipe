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

#pragma once

#include <atomic>
#include <cassert>
#include <memory>
#include <numeric>
#include <string>

#include <vector>
#include "time_utils.hpp"

#include <algorithm>
#include "Backend.hpp"
#include "dict.hpp"
#include "params.hpp"
#include "reflect.h"
#include "state_event.hpp"
#include "threadsafe_queue.hpp"
namespace ipipe {

/**
 * @brief 提供多实例功能，与 Schedule
 * SortSchedule 等配合使用，形成完整的单节点调度功能。
 */
class Instances : public Backend {
 public:
  /**
   * @brief
   *
   * @param Instances::backend
   * 实现backend消费逻辑的子后端。默认为 InstanceHandler.
   * @param instance_num 实例数目。
   * @param[out] _state_event std::shared_ptr< @ref StateEvents>,
   * 消费状态，用于通知实例使用状态。如果dict_config非空，输出到 dict_config
   * 中。
   * @param[out] _batched_queue ThreadSafeQueue<std::vector<dict>>*,
   * 存放batch数据的队列。写入Instances::backend的输入参数 dict_config。
   * @param[out] _independent_thread_index 从 0 到 instance_num - 1，
   * 写入Instances::backend->init的输入参数config中。
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
    while (!batched_queue_.empty()) {  // todo state empty full, invalid, half StateEvent
      std::this_thread::yield();
    }

    while (!inputs.empty()) {
      int index = 0;
      auto input_true = split_inputs(inputs, index);
      assert(!input_true.empty());  // todo check at init

      batched_queue_.Push(input_true);
      assert(index < backends_.size());

      backends_[index]->forward(input_true);
    }
  }

  /// 子Backend的上限最大值 @todo use RangeMerger Backend
  uint32_t max() const override {
    std::vector<uint32_t> maxs;
    for (std::size_t i = 0; i < backends_.size(); ++i) {
      maxs.push_back(backends_[i]->max());
    }
    return *std::max_element(maxs.begin(), maxs.end());
  }

  /// 子Backend的下限最小值 @todo use RangeMerger Backend
  uint32_t min() const override {
    std::vector<uint32_t> mins;
    for (std::size_t i = 0; i < backends_.size(); ++i) {
      mins.push_back(backends_[i]->min());
    }
    return *std::min_element(mins.begin(), mins.end());
  }

#ifndef DOXYGEN_SHOULD_SKIP_THIS
  // 先销毁消费线程， 确保销毁顺序
  ~Instances() { backends_.clear(); }
#endif

 private:
  std::unique_ptr<Params> params_;
  std::vector<std::unique_ptr<Backend>> backends_;
  ThreadSafeQueue<std::vector<dict>> batched_queue_;
  uint32_t instance_num_;
  std::shared_ptr<StateEvents> state_;

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
      if (size_of_input >= backends_[item]->min()) {
        return item;
      }
    }
    return -1;
  }

  std::vector<dict> split_inputs(std::vector<dict>& inputs, int& index) {
    assert(!inputs.empty());
    index = get_best_match(inputs.size());
    assert(index >= 0);
    assert(inputs.size() >= backends_[index]->min());

    if (inputs.size() <= backends_[index]->max()) {
      std::vector<dict> out;
      std::swap(out, inputs);
      return out;
    } else {
      std::vector<dict> out(inputs.begin(), inputs.begin() + backends_[index]->max());
      inputs = std::vector<dict>(inputs.begin() + backends_[index]->max(), inputs.end());
      return out;
    }
  }

  std::vector<std::size_t> sorted_max_;
};
}  // namespace ipipe