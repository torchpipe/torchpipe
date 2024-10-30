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
#include "Backend.hpp"
#include "dict.hpp"
#include <torch/torch.h>

namespace ipipe {
/**
 * @brief @ref TensorrtTensor 提供的后处理操作的扩展，
 可继承并实现自定义后处理。
 *
 *
 * **使用示例**
  ```
  # 这里仅以toml配置文件方式展示Remove的使用，其他方式使用同理：
  [resnet]
  backend = "Sequential[TensorrtTensor, SyncTensor]"
  max=4
  model = "/app/src/models/ex_model.onnx"
  instance_num = 2
  postprocessor = "resnet_post"
  IPIPE_REGISTER(PostProcessor<torch::Tensor>, YOUR_POST_IMPLEMENTION,
 "resnet_post")
  ```
 *
 */
template <typename T>
class PostProcessor {
 public:
  /**
   * @brief 初始化函数
   *
   */
  virtual bool init(const std::unordered_map<std::string, std::string>& /*config*/,
                    dict /*dict_config*/) {
    return true;
  };
  virtual void forward(std::vector<T> net_outputs, std::vector<dict> inputs,
                       const std::vector<T>& net_inputs) {
    throw std::runtime_error("Not implemented");
  };

#ifndef DOXYGEN_SHOULD_SKIP_THIS
  virtual ~PostProcessor() = default;
#endif
};

// 声明特化
template <>
void PostProcessor<torch::Tensor>::forward(std::vector<torch::Tensor> net_outputs,
                                           std::vector<dict> inputs,
                                           const std::vector<torch::Tensor>& net_inputs);

/**
 * @brief 自定义前处理，由 @ref TensorrtTensor 后端扩展的功能
 * @ref SingleConcatPreprocess @ref MultipleConcatPreprocess.
 */
template <typename T>
class PreProcessor {
 public:
  /**
   * @brief 与后端类似的初始化功能
   */
  virtual bool init(const std::unordered_map<std::string, std::string>& /*config*/,
                    dict /*dict_config*/) {
    return true;
  };
  /**
   * @brief 返回网络输入数据。
   */
  virtual std::vector<T> forward(const std::vector<dict>& inputs) = 0;
  virtual ~PreProcessor() = default;
};
}  // namespace ipipe