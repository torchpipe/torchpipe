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
#ifdef DEBUG
#include "base_logging.hpp"
#endif
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
  /**
   * @brief 前向函数，默认实现是拆分batch后将网络输出的结果赋值给@ref
   * TASK_RESULT_KEY.
   * @param net_outputs
   * 按照batch返回的Backend结果，以常用的Backend::TensorTensor举例，该参数就是模型的输出，
   * 该参数为vector类型，代表模型多个分支的输出结果，如果模型只有一个分支，直接用net_output[0]即可拿到结果，
   * 如果有多个分支，可以使用net_output[1],net_output[2]等。
   *
   * @note
   * 可通过如下代码，将一个batch的数据直接从gpu复制到cpu(以返回torch::Tensor举例)。
   * ```
   * torch::Tensor cls_result = net_outputs[0].to(torch::kCPU);
   * ```
   * @param inputs 数据字典，注意这里是按照batch的。
   *
   * @param net_inputs 网络输入。
   * @remark 如果重写了这个类，注意返回参数需要有@ref TASK_RESULT_KEY.
   * 否则认为该节点发生了异常，结果不可用。
   *
   */
  virtual void forward(std::vector<T> net_outputs, std::vector<dict> inputs,
                       const std::vector<T>& net_inputs) {
    SPDLOG_DEBUG("PostProcessor input size:{}, request size:{}, output[0] size[0]:{}",
                 inputs.size(), get_request_size(inputs), net_outputs[0].sizes()[0]);
    if (inputs.size() == 1) {
      if (net_outputs.size() == 1)
        (*inputs[0])[TASK_RESULT_KEY] = net_outputs[0];
      else
        (*inputs[0])[TASK_RESULT_KEY] = net_outputs;
      return;
    } else if (net_outputs[0].sizes()[0] > inputs.size()) {
      std::vector<uint32_t> shapes{0};
      for (const auto& item : inputs) {
        shapes.push_back(get_request_size(item));
      }
      IPIPE_ASSERT(std::accumulate(shapes.begin(), shapes.end(), 0) == net_outputs[0].sizes()[0]);
      // 累加
      std::partial_sum(shapes.begin(), shapes.end(), shapes.begin());

      for (std::size_t i = 0; i < inputs.size(); ++i) {
        std::vector<T> single_result;
        for (const auto& item : net_outputs) {
          single_result.push_back(item.index({torch::indexing::Slice(shapes[i], shapes[i + 1])}));
        }
        if (single_result.size() == 1) {
          (*inputs[i])[TASK_RESULT_KEY] = single_result[0];  // 返回torch::Tensor
        } else
          (*inputs[i])[TASK_RESULT_KEY] = single_result;  // 返回std::vector<torch::Tensor>
      }
    } else {
      for (std::size_t i = 0; i < inputs.size(); ++i) {
        std::vector<T> single_result;
        for (const auto& item : net_outputs) {
          single_result.push_back(item[i].unsqueeze(0));
        }
        if (single_result.size() == 1) {
          (*inputs[i])[TASK_RESULT_KEY] = single_result[0];
        } else
          (*inputs[i])[TASK_RESULT_KEY] = single_result;
      }
    }
  }

#ifndef DOXYGEN_SHOULD_SKIP_THIS
  virtual ~PostProcessor() = default;
#endif
};

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