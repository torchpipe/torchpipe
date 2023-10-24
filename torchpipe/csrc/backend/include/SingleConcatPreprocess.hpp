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

#include <ATen/ATen.h>

#include "params.hpp"
#include "prepost.hpp"

namespace ipipe {

/**
 * @brief preprocess for inference. It runs 'to_cuda', 'unsqueeze', 'cat' and
 * 'permute'(if tensor.size(-1) is 1 or 3) one by one.
 */
class SingleConcatPreprocess : public PreProcessor<at::Tensor> {
 public:
  virtual bool init(const std::unordered_map<std::string, std::string>& /*config*/,
                    dict /*dict_config*/) override;
  /**
   * @brief forward function. It turns raw_inputs into one single batched
   * Tensor-Group.
   *
   * @param raw_inputs input data. 'data' of raw_inputs should be of type
   * 'at::Tensor'.
   *
   * @return std::vector<at::Tensor> usually concated tensor(s).
   */
  std::vector<at::Tensor> forward(const std::vector<dict>& raw_inputs);

 private:
  std::vector<std::vector<int>> max_value_;
  std::vector<std::vector<int>> min_value_;
};

}  // namespace ipipe