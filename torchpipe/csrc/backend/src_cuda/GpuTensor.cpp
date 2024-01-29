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
namespace ipipe {

/**
 * @brief cpu->gpu
 */
class GpuTensor : public SingleBackend {
 public:
  /**
   * @brief cpu->gpu
   * @param TASK_RESULT_KEY input[TASK_RESULT_KEY] = input[TASK_DATA_KEY].cuda()
   */
  virtual void forward(dict input_dict) override {
    auto& input = *input_dict;
    if (input[TASK_DATA_KEY].type() != typeid(at::Tensor)) {
      SPDLOG_ERROR("GpuTensor: at::Tensor needed; error input type: " +
                   std::string(input[TASK_DATA_KEY].type().name()));
      throw std::runtime_error("GpuTensor: at::Tensor needed; error input type: " +
                               std::string(input[TASK_DATA_KEY].type().name()));
    }
    auto input_tensor = any_cast<at::Tensor>(input[TASK_DATA_KEY]);

    if (!is_cpu_tensor(input_tensor)) {
      SPDLOG_ERROR("input_tensor should be cpu tensor");
      throw std::runtime_error("input_tensor should be cpu tensor");
    }

    input[TASK_RESULT_KEY] = input_tensor.cuda();
  }

 private:
};

IPIPE_REGISTER(Backend, GpuTensor, "GpuTensor");

/**
 * @brief gpu->cpu
 */
class CpuTensor : public SingleBackend {
 public:
  /**
   * @brief cpu->gpu
   * @param TASK_RESULT_KEY input[TASK_RESULT_KEY] = input[TASK_DATA_KEY].cuda()
   */
  virtual void forward(dict input_dict) override {
    auto& input = *input_dict;
    if (input[TASK_DATA_KEY].type() != typeid(at::Tensor)) {
      SPDLOG_ERROR("GpuTensor: at::Tensor needed; error input type: " +
                   std::string(input[TASK_DATA_KEY].type().name()));
      throw std::runtime_error("GpuTensor: at::Tensor needed; error input type: " +
                               std::string(input[TASK_DATA_KEY].type().name()));
    }
    auto input_tensor = any_cast<at::Tensor>(input[TASK_DATA_KEY]);

    if (!input_tensor.is_cuda()) {
      SPDLOG_ERROR("input_tensor should be gpu tensor");
      throw std::runtime_error("input_tensor should be gpu tensor");
    }

    input[TASK_RESULT_KEY] = input_tensor.cpu();
  }

 private:
};

IPIPE_REGISTER(Backend, CpuTensor, "CpuTensor");

/**
 * @brief to float
 */
class FloatTensor : public SingleBackend {
 public:
  /**
   */

  /**
   * @brief to float
   * @param TASK_RESULT_KEY input[TASK_RESULT_KEY] = input[TASK_DATA_KEY].float()
   */
  virtual void forward(dict input_dict) override {
    auto& input = *input_dict;
    if (input[TASK_DATA_KEY].type() != typeid(at::Tensor)) {
      SPDLOG_ERROR("FloatTensor: at::Tensor needed; error input type: " +
                   std::string(input[TASK_DATA_KEY].type().name()));
      throw std::runtime_error("FloatTensor: at::Tensor needed; error input type: " +
                               std::string(input[TASK_DATA_KEY].type().name()));
    }
    auto input_tensor = any_cast<at::Tensor>(input[TASK_DATA_KEY]);

    input[TASK_RESULT_KEY] = input_tensor.to(at::kFloat);
  }

 private:
};

IPIPE_REGISTER(Backend, FloatTensor, "FloatTensor");

}  // namespace ipipe