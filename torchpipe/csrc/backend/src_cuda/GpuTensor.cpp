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

#include <torch/torch.h>
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
    if (input[TASK_DATA_KEY].type() != typeid(torch::Tensor)) {
      SPDLOG_ERROR("GpuTensor: torch::Tensor needed; error input type: " +
                   std::string(input[TASK_DATA_KEY].type().name()));
      throw std::runtime_error("GpuTensor: torch::Tensor needed; error input type: " +
                               std::string(input[TASK_DATA_KEY].type().name()));
    }
    auto input_tensor = any_cast<torch::Tensor>(input[TASK_DATA_KEY]);

    if (input_tensor.is_cpu()) {
      // SPDLOG_ERROR("input_tensor should be cpu tensor");
      // throw std::runtime_error("input_tensor should be cpu tensor");
      input[TASK_RESULT_KEY] = input_tensor;
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

    if (input[TASK_DATA_KEY].type() == typeid(torch::Tensor)) {
      torch::Tensor input_tensor = any_cast<torch::Tensor>(input[TASK_DATA_KEY]);
      if (!input_tensor.is_cuda()) {
        SPDLOG_ERROR("input_tensor should be gpu tensor");
        throw std::runtime_error("input_tensor should be gpu tensor");
      }

      input[TASK_RESULT_KEY] = input_tensor.cpu();
    } else if (input[TASK_DATA_KEY].type() == typeid(std::vector<torch::Tensor>)) {
      std::vector<torch::Tensor> input_tensor =
          any_cast<std::vector<torch::Tensor>>(input[TASK_DATA_KEY]);
      for (auto& item : input_tensor) {
        if (item.is_cuda()) {
          item = item.cpu();
        } else {
          SPDLOG_ERROR("input_tensor should be gpu tensor");
          throw std::runtime_error("input_tensor should be gpu tensor");
        }
      }
      input[TASK_RESULT_KEY] = input_tensor;
    } else {
      SPDLOG_ERROR(
          "CpuTensor: torch::Tensor/std::vector<torch::Tensor> needed; error input type: " +
          std::string(input[TASK_DATA_KEY].type().name()));
      throw std::runtime_error(
          "CpuTensor: torch::Tensor/std::vector<torch::Tensor> needed; error input type: " +
          std::string(input[TASK_DATA_KEY].type().name()));
    }
  }
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
    if (input[TASK_DATA_KEY].type() != typeid(torch::Tensor)) {
      SPDLOG_ERROR("FloatTensor: torch::Tensor needed; error input type: " +
                   std::string(input[TASK_DATA_KEY].type().name()));
      throw std::runtime_error("FloatTensor: torch::Tensor needed; error input type: " +
                               std::string(input[TASK_DATA_KEY].type().name()));
    }
    auto input_tensor = any_cast<torch::Tensor>(input[TASK_DATA_KEY]);

    input[TASK_RESULT_KEY] = input_tensor.to(torch::kFloat);
  }

 private:
};

IPIPE_REGISTER(Backend, FloatTensor, "FloatTensor");

/**
 * @brief to float
 */
class LongItemTensor : public SingleBackend {
 public:
  /**
   */

  /**
   * @brief to float
   * @param TASK_RESULT_KEY input[TASK_RESULT_KEY] = input[TASK_DATA_KEY].float()
   */
  virtual void forward(dict input_dict) override {
    auto& input = *input_dict;
    if (input[TASK_DATA_KEY].type() != typeid(torch::Tensor)) {
      SPDLOG_ERROR("LongItemTensor: torch::Tensor needed; error input type: " +
                   std::string(input[TASK_DATA_KEY].type().name()));
      throw std::runtime_error("LongItemTensor: torch::Tensor needed; error input type: " +
                               std::string(input[TASK_DATA_KEY].type().name()));
    }
    auto input_tensor = any_cast<torch::Tensor>(input[TASK_DATA_KEY]);
    if (input_tensor.numel() != 1) {
      SPDLOG_ERROR("LongItemTensor: numel should be 1; error input numel: " +
                   std::to_string(input_tensor.numel()));
      throw std::runtime_error("LongItemTensor: numel should be 1; error input numel: " +
                               std::to_string(input_tensor.numel()));
    }

    input[TASK_RESULT_KEY] = input_tensor.item<long>();
  }
};

IPIPE_REGISTER(Backend, LongItemTensor, "LongItemTensor");

class NCHWTensor : public SingleBackend {
 public:
  /**
   */

  /**
   * @brief to float
   * @param TASK_RESULT_KEY input[TASK_RESULT_KEY] = input[TASK_DATA_KEY].float()
   */
  virtual void forward(dict input_dict) override {
    auto& input = *input_dict;
    if (input[TASK_DATA_KEY].type() != typeid(torch::Tensor)) {
      SPDLOG_ERROR("FloatTensor: torch::Tensor needed; error input type: " +
                   std::string(input[TASK_DATA_KEY].type().name()));
      throw std::runtime_error("FloatTensor: torch::Tensor needed; error input type: " +
                               std::string(input[TASK_DATA_KEY].type().name()));
    }
    auto input_tensor = any_cast<torch::Tensor>(input[TASK_DATA_KEY]);

    auto tensor = tensor2nchw(input_tensor);
    // if (tensor.scale_type() != torch::kFloat) {
    //   tensor = tensor.float();
    // }
    // if (!tensor.is_contiguous()) {
    //   tensor = tensor.contiguous();
    // }
    input[TASK_RESULT_KEY] = tensor;
  }
};

IPIPE_REGISTER(Backend, NCHWTensor, "NCHWTensor");

/**
 * @brief to nchw
 */
class FloatNCHWTensor : public SingleBackend {
 public:
  /**
   */

  /**
   * @brief to float
   * @param TASK_RESULT_KEY input[TASK_RESULT_KEY] = input[TASK_DATA_KEY].float()
   */
  virtual void forward(dict input_dict) override {
    auto& input = *input_dict;
    if (input[TASK_DATA_KEY].type() != typeid(torch::Tensor)) {
      SPDLOG_ERROR("FloatTensor: torch::Tensor needed; error input type: " +
                   std::string(input[TASK_DATA_KEY].type().name()));
      throw std::runtime_error("FloatTensor: torch::Tensor needed; error input type: " +
                               std::string(input[TASK_DATA_KEY].type().name()));
    }
    auto input_tensor = any_cast<torch::Tensor>(input[TASK_DATA_KEY]);

    auto tensor = tensor2nchw(input_tensor);
    if (tensor.scalar_type() != torch::kFloat) {
      tensor = tensor.to(torch::kFloat);
    }
    if (!tensor.is_contiguous()) {
      tensor = tensor.contiguous();
    }
    input[TASK_RESULT_KEY] = tensor;
  }
};

IPIPE_REGISTER(Backend, FloatNCHWTensor, "FloatNCHWTensor");
}  // namespace ipipe