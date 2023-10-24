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
#include "dict.hpp"
#include "filter.hpp"

#include <memory>
#include <vector>

namespace ipipe {
class Params;
/**
 * @brief 一系列后端的串行容器，容器内的后端共用一个调度，采用串行执行的方式，顺序执行。简写为 `S`.
 * @note 此容器前向时会临时保存 TASK_DATA_KEY，输出时，将还原 TASK_DATA_KEY.
 *  使用示例:
    ```toml
    # 这里仅以toml配置文件方式展示Sequential的使用，其他方式使用同理：

    [resnet]
    backend = "Sequential[TensorrtTensor, SyncTensor]"
    max=4
    model = "/app/src/models/ex_model.onnx" # or resnet18_merge
    instance_num = 2
    next = "postprocess"
    ```
 *
 */

class Sequential final : public Backend {
 public:
  /**
   * @brief 初始化函数。
   * @param Sequential::backend 子后端名称，多个后端，逗号分开。
   * @remark
   * 1. 子后端的初始化将按照 **倒序串联** 执行;
   * 2. 此容器支持将括号复合语法展开。利用  @ref brackets_split
   * 函数将其展开，展开方式如下：
   * - backend = B[C]        =>     {backend=B, B::backend=C}
   * - backend = D           =>     {backend=D}
   * - backend = B[E[Z1,Z2]] =>     {backend=B, B::backend=E[Z1,Z2]}
   *
   * @return true or false
   *
   *
   */
  virtual bool init(const std::unordered_map<std::string, std::string>&, dict) override;

  /**
   * @brief 按照顺序调用子后端
   * @note 顺序调用过程中，会做swap
   * filter操作：将TASK_RESULT_KEY赋值给TASK_DATA_KEY，并删除原来的TASK_DATA_KEY。
   */

  virtual void forward(const std::vector<dict>&) override;

  /**
   * @brief 输入区间最大值。
   * @remark  输入范围计算规则：
   * 1. 如果子后端的输入区间为[1,1], 提升为[1, UINT32_MAX]
   * 2. 求得子后端的输入区间的并集[x,y]
   * 3. 如果 y==UINT32_MAX, 将y设为1.
   * 4. 检查是否满足 x<=y.
   *
   *
   */
  virtual uint32_t max() const { return max_; }
  /// @brief 输入区间最小值。
  virtual uint32_t min() const { return min_; }

 private:
  std::unique_ptr<Params> params_;
  std::vector<std::unique_ptr<Backend>> engines_;
  std::vector<std::unique_ptr<Filter>> filters_;
  std::vector<std::string> engine_names_;
  unsigned min_{1};
  unsigned max_{UINT32_MAX};
  std::string register_name_;
};
}  // namespace ipipe
