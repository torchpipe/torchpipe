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

#include <ATen/ATen.h>
#include <torch/script.h>
#include "Backend.hpp"
#include "dict.hpp"
#include "params.hpp"
#include "Backend.hpp"
#include "prepost.hpp"

namespace ipipe {

/**
 * @brief 加载torchscript模型并且前向推理
 */
class TorchScriptTensor : public Backend {
 public:
  /**
   * @param model 必须传入，TorchScript模型路径
   */
  virtual bool init(const std::unordered_map<std::string, std::string>&, dict) override;

  /**
   * @param  TASK_DATA_KEY 类型为 at::Tensor
   */
  virtual void forward(const std::vector<dict>&) override;

  virtual uint32_t max() const override { return max_; }

 private:
  torch::jit::script::Module module_;
  std::unique_ptr<Params> params_;
  std::unique_ptr<PreProcessor<at::Tensor>> preprocessor_;
  std::unique_ptr<PostProcessor<at::Tensor>> postprocessor_;

  int num_inputs_{0};
  u_int32_t max_{1};
};

}  // namespace ipipe
