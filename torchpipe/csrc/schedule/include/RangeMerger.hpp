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

#include "Backend.hpp"
#include <numeric>

#include <memory>
#include "params.hpp"

namespace ipipe {
/**
* @brief
将固定输入范围合并到动态输入范围， 比如将[1,1] 和 [4,4]
合并为[1,4]。调用链路为
``BaselineSchedule[RangeMerger[MultipleInstances[InstanceHandler[backend]]]]``
@warning 不直接作用于计算后端。作用于调度后端。参考 BaselineSchedule.
*
* **使用示例:**
  ```
  # 这里仅以单节点toml配置文件方式展示使用，其他方式使用同理：
  [resnet]
  #多节点
  #"PipelineV3::backend"=BaselineSchedule
  backend = "SyncTensor[TensorrtTensor]"
  max="1&4"
  min="1&4"
  model = "batch1.onnx&batch4.onnx" # or resnet18_merge
  instance_num = "1&2"
  next = "postprocess"
  ```
*
*/
class RangeMerger final : public Backend {
 public:
  /**
   * @brief 配置使用 ``&``
   * 分组，根据最长的一组配置确定组的数目。不足的配置自动以最后一位补齐。每组生成一个子后端。
   * @param RangeMerger::backend 子后端。默认为 @ref MultipleInstances.
   * @note 子后端输入范围最最小值需要为1，否则初始化失败。
   *
   */
  virtual bool init(const std::unordered_map<std::string, std::string>&, dict) override;

  /**
   * @brief 根据输入数量自动把数据分配到合适的子后端中执行。
   */
  virtual void forward(const std::vector<dict>&) override;

  /**
   * @brief 子后端输入范围最大值。
   *
   *
   */
  virtual uint32_t max() const { return max_; }
  /// @brief 1.
  virtual uint32_t min() const { return 1; }

 private:
  template <typename T>
  std::vector<std::size_t> sort_vector(const std::vector<T>& v) {
    std::vector<std::size_t> index(v.size());
    std::iota(index.begin(), index.end(), 0);

    std::stable_sort(index.begin(), index.end(),
                     [&v](std::size_t i1, std::size_t i2) { return v[i1]->max() > v[i2]->max(); });

    return index;
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

  int get_best_match(std::size_t size_of_input) {
    for (auto item : sorted_max_) {
      if (size_of_input >= backends_[item]->min()) {
        return item;
      }
    }
    return -1;
  }

 private:
  std::unique_ptr<Params> params_;
  uint32_t max_{0};
  std::vector<std::unique_ptr<Backend>> backends_;
  std::vector<std::size_t> sorted_max_;
};

}  // namespace ipipe
