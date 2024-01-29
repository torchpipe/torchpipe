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

#include "MultipleConcatPreprocess.hpp"

#include <ATen/ATen.h>

// #include <fstream>
#include <memory>
#include <string>

#include "base_logging.hpp"

#include "prepost.hpp"
#include "reflect.h"
#include "torch_utils.hpp"

namespace ipipe {

bool MultipleConcatPreprocess::init(const std::unordered_map<std::string, std::string>& config,
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
  if (min_value_.size() == 1 || max_value_.size() == 1 || min_value_.size() != max_value_.size()) {
    throw std::runtime_error("min_value_ == 1 || max_value_ == 1 || min_value_ != max_value_");
  }
  for (std::size_t i = 0; i < min_value_.size(); ++i) {
    if (min_value_[i].size() != max_value_[i].size() || (min_value_[i].size() <= 1)) {
      throw std::runtime_error(
          "min_value_[i].size() != max_value_[i].size() || (min_value_[i].size() <= 1)");
    }
  }

  return true;
}

std::vector<at::Tensor> MultipleConcatPreprocess::forward(const std::vector<dict>& raw_inputs) {
  std::vector<std::vector<at::Tensor>> resized_inputs;

  bool may_need_quick_cat = true;
  for (std::size_t index = 0; index < raw_inputs.size(); ++index) {
    auto item = raw_inputs[index];
    auto iter_data = item->find(TASK_DATA_KEY);
    if (iter_data == item->end()) {
      throw std::runtime_error("MultipleConcatPreprocess: data not exists.");
    }

    if (iter_data->second.type() != typeid(std::vector<at::Tensor>)) {
      throw std::runtime_error(
          "MultipleConcatPreprocess: data type must be std::vector<at::Tensor>.");
    }
    std::vector<at::Tensor> net_inputs = any_cast<std::vector<at::Tensor>>(iter_data->second);
    if (!min_value_.empty() && net_inputs.size() != min_value_.size()) {
      SPDLOG_ERROR("input.size == {}, min_value_.size == {}", net_inputs.size(), min_value_.size());
      throw std::runtime_error("shape not match");
    }
    for (std::size_t i = 0; i < net_inputs.size(); ++i) {
      auto& net_input = net_inputs[i];

      if (is_cpu_tensor(net_input)) {
        may_need_quick_cat = false;
        //  注意：前处理在某些特殊情况下在cpu上可能变得特别慢，
        net_input = net_input.to(at::kCUDA, net_input.dtype(), /* non_blocking =*/true, false);
      }

      assert(net_input.is_cuda());
      if (!min_value_.empty()) {
        bool need_permute = false;
        net_input = tensor_permute(net_input, min_value_[i], max_value_[i], need_permute);
        if (need_permute || !net_input.is_contiguous()) {
          may_need_quick_cat = false;
        }
      }
    }

    resized_inputs.push_back(net_inputs);
  }
  int num_input = resized_inputs[0].size();
  if (resized_inputs.size() > 1) {
    for (const auto& inputs : resized_inputs) {
      for (std::size_t j = 0; j < inputs.size(); ++j) {
        if (inputs[j].sizes() != resized_inputs[0][j].sizes()) {
          std::stringstream s_err;
          s_err << inputs[j].sizes() << " != " << resized_inputs[0][j].sizes()
                << ", MultipleConcatPreprocess: concat failed";
          SPDLOG_ERROR(s_err.str().c_str());
          return {};
        }
      }
    }
  }

  std::vector<at::Tensor> result;
  std::size_t size_input = 0;
  for (int j = 0; j < num_input; ++j) {
    std::vector<at::Tensor> data;
    for (const auto& inputs : resized_inputs) {
      data.push_back(inputs[j]);
    }
    at::Tensor true_input;
    if (data.size() > 1) {
      if (may_need_quick_cat) {
        true_input = try_quick_cat(data);
      } else {
        true_input = at::cat(data, 0);
      }
    } else if (data.empty()) {
      throw std::runtime_error("ConcatPreprocess: data is empty.");
    } else {
      true_input = data[0];
    }

    result.emplace_back(true_input);
    // if (size_input == 0)
    //   size_input = result.back().sizes().size();
    // else if (size_input != result.back().sizes().size()) {
    //   SPDLOG_ERROR("shape of input tensors not match: " + std::to_string(size_input) +
    //                "!=" + std::to_string(result.back().sizes().size()));
    //   return {};
    // }
  }
  return result;
}

IPIPE_REGISTER(PreProcessor<at::Tensor>, MultipleConcatPreprocess, "MultipleConcatPreprocess");

}  // namespace ipipe

// https://github.com/NVIDIA/Torch-TensorRT/blob/3a98a8b198a071e622c43283caea7416fe8a8a1a/core/runtime/register_trt_op.cpp
