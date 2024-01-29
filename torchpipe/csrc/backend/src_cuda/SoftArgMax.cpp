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

#include <ATen/ATen.h>
#include <fstream>
#include "base_logging.hpp"
#include "reflect.h"
#include "dict.hpp"
#include "params.hpp"
#include "torch_utils.hpp"
#include <torch/torch.h>
namespace ipipe {

/**
 * @brief cpu->gpu
 */
class SoftmaxArgMaxTensor : public Backend {
 public:
  /**
   * @brief cpu->gpu
   * @param TASK_RESULT_KEY input[TASK_RESULT_KEY] = input[TASK_DATA_KEY].cuda()
   */
  virtual void forward(const std::vector<dict>& input_dicts) override {
    auto& input = *input_dicts[0];
    if (input[TASK_DATA_KEY].type() != typeid(at::Tensor)) {
      SPDLOG_ERROR("SoftmaxArgMaxTensor: at::Tensor needed; error input type: " +
                   std::string(input[TASK_DATA_KEY].type().name()));
      throw std::runtime_error("SoftmaxArgMaxTensor: at::Tensor needed; error input type: " +
                               std::string(input[TASK_DATA_KEY].type().name()));
    }
    auto input_tensor = any_cast<at::Tensor>(input[TASK_DATA_KEY]);

    // IPIPE_ASSERT(input_tensor.sizes().size() == 2);
    at::Tensor output = input_tensor.softmax(-1);
    auto max_values_and_indices = torch::max(output, -1);

    torch::Tensor max_values = std::get<0>(max_values_and_indices).cpu();
    torch::Tensor max_indices = std::get<1>(max_values_and_indices).cpu();

    // for (std::size_t i = 0; i < 1; ++i) {
    float max_score = max_values.item<float>();
    long argmax = max_indices.item<long>();
    if (argmax > 10000) {
    }

    input["score"] = max_score;
    input["class"] = static_cast<int>(argmax);
    input[TASK_RESULT_KEY] = static_cast<int>(argmax);
    // }
  }
};

IPIPE_REGISTER(Backend, SoftmaxArgMaxTensor, "SoftmaxArgMaxTensor");

/**
 * @brief cpu->gpu
 */
class ArgMaxTensor : public Backend {
 public:
  /**
   * @brief cpu->gpu
   * @param TASK_RESULT_KEY input[TASK_RESULT_KEY] = input[TASK_DATA_KEY].cuda()
   */
  virtual void forward(const std::vector<dict>& input_dicts) override {
    auto& input = *input_dicts[0];
    if (input[TASK_DATA_KEY].type() != typeid(at::Tensor)) {
      SPDLOG_ERROR("ArgMaxTensor: at::Tensor needed; error input type: " +
                   std::string(input[TASK_DATA_KEY].type().name()));
      throw std::runtime_error("ArgMaxTensor: at::Tensor needed; error input type: " +
                               std::string(input[TASK_DATA_KEY].type().name()));
    }
    auto input_tensor = any_cast<at::Tensor>(input[TASK_DATA_KEY]);

    // IPIPE_ASSERT(input_tensor.sizes().size() == 2);
    at::Tensor output = input_tensor.softmax(-1);
    auto max_values_and_indices = torch::max(output, -1);

    torch::Tensor max_values = std::get<0>(max_values_and_indices).cpu();
    torch::Tensor max_indices = std::get<1>(max_values_and_indices).cpu();

    // for (std::size_t i = 0; i < 1; ++i) {
    float max_score = max_values.item<float>();
    long argmax = max_indices.item<long>();
    if (argmax > 10000) {
    }

    input["score"] = max_score;
    input["class"] = static_cast<int>(argmax);
    input[TASK_RESULT_KEY] = static_cast<int>(argmax);
    // }
  }
};

IPIPE_REGISTER(Backend, ArgMaxTensor, "ArgMaxTensor");

/**
 * @brief cpu->gpu
 */
class Tensor2Vector : public Backend {
 public:
  /**
   * @brief cpu->gpu
   * @param TASK_RESULT_KEY input[TASK_RESULT_KEY] = input[TASK_DATA_KEY].cuda()
   */
  virtual void forward(const std::vector<dict>& input_dicts) override {
    auto& input = *input_dicts[0];
    if (input[TASK_DATA_KEY].type() != typeid(at::Tensor)) {
      SPDLOG_ERROR("Tensor2Vector: at::Tensor needed; error input type: " +
                   std::string(input[TASK_DATA_KEY].type().name()));
      throw std::runtime_error("Tensor2Vector: at::Tensor needed; error input type: " +
                               std::string(input[TASK_DATA_KEY].type().name()));
    }
    auto input_tensor = any_cast<at::Tensor>(input[TASK_DATA_KEY]);
    if (input_tensor.is_cuda()) {
      input_tensor = input_tensor.cpu();
    }
    if (!input_tensor.is_contiguous()) {
      input_tensor = input_tensor.contiguous();
    }

    IPIPE_ASSERT(input_tensor.scalar_type() == at::kFloat);

    float* data = input_tensor.data_ptr<float>();
    int numel = input_tensor.numel();

    std::vector<float> result = std::vector<float>(data, data + numel);

    input[TASK_RESULT_KEY] = result;
    // }
  }
};

IPIPE_REGISTER(Backend, Tensor2Vector, "Tensor2Vector");

}  // namespace ipipe