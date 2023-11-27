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

#include "SingleConcatPreprocess.hpp"

#include <ATen/ATen.h>

#include <memory>
#include <string>

#include "base_logging.hpp"
#include "torch_utils.hpp"
#include "prepost.hpp"
#include "reflect.h"

namespace ipipe {

bool SingleConcatPreprocess::init(const std::unordered_map<std::string, std::string>& config,
                                  dict dict_config) {
  if (dict_config) {
    auto iter = dict_config->find("max");
    if (iter != dict_config->end()) {
      max_value_ = any_cast<std::vector<std::vector<int>>>(iter->second);
    }
    iter = dict_config->find("min");
    if (iter != dict_config->end()) {
      min_value_ = any_cast<std::vector<std::vector<int>>>(iter->second);
    }
  }
  if (min_value_.size() >= 2 || max_value_.size() >= 2 || min_value_.size() != max_value_.size()) {
    throw std::runtime_error("min_value_ >= 2 || max_value_ >= 2 || min_value_ != max_value_");
  }
  if (!min_value_.empty()) {
    if (min_value_[0].size() != max_value_[0].size() || (min_value_[0].empty())) {
      throw std::runtime_error(
          "min_value_[0].size() != max_value_[0].size() || (min_value_[0].size() <= 1)");
    }
  }

  return true;
}

std::vector<at::Tensor> SingleConcatPreprocess::forward(const std::vector<dict>& raw_inputs) {
  std::vector<at::Tensor> resized_inputs;
  for (std::size_t index = 0; index < raw_inputs.size(); ++index) {
    auto item = raw_inputs[index];
    auto iter_data = item->find(TASK_DATA_KEY);
    if (iter_data == item->end()) {
      throw std::runtime_error("SingleConcatPreprocess: data not exists.");
    }

    if (iter_data->second.type() != typeid(at::Tensor)) {
      throw std::runtime_error(
          "SingleConcatPreprocess: data type must be tensor. your input type is " +
          c10::demangle(iter_data->second.type().name()));
    }
    at::Tensor net_input = any_cast<at::Tensor>(iter_data->second);

    if (is_cpu_tensor(net_input)) {
      //  注意：前处理在某些特殊情况下在cpu上可能变得特别慢，
      //  所以这里我们将他拷贝到cuda上； 建议overlead computing 和 transform
      net_input = net_input.to(at::kCUDA, net_input.dtype(), /* non_blocking =*/false, false);
    }
    if (!min_value_.empty()) net_input = tensor_permute(net_input, min_value_[0], max_value_[0]);
    resized_inputs.push_back(net_input);
  }
  assert(!resized_inputs.empty());

  auto true_input = at::cat(resized_inputs, 0);

  if (raw_inputs.size() > true_input.size(0)) {
    SPDLOG_ERROR("wired batchsize: need {}, get {}", raw_inputs.size(), true_input.size(0));
    return {};
  }
  return {true_input};
}

IPIPE_REGISTER(PreProcessor<at::Tensor>, SingleConcatPreprocess, "SingleConcatPreprocess");

}  // namespace ipipe

// https://github.com/NVIDIA/Torch-TensorRT/blob/3a98a8b198a071e622c43283caea7416fe8a8a1a/core/runtime/register_trt_op.cpp
