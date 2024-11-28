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

#include <torch/torch.h>

#include <memory>
#include <string>

#include <NvInferRuntime.h>
#include "Backend.hpp"
#include "dict.hpp"

// #include "tensorrt_utils.hpp"

namespace ipipe {
class CudaEngineWithRuntime;
struct OnnxParams {
  std::string precision;
  std::set<std::string> precision_fp32;
  std::set<std::string> precision_fp16;
  std::set<std::string> precision_output_fp32;
  std::set<std::string> precision_output_fp16;
  std::string timecache;
  size_t max_workspace_size;
  std::string allocator;
  std::string log_level;
  std::vector<int> input_reorder;
  int force_layer_norm_pattern_fp32;
  int weight_budget_percentage;
};

std::shared_ptr<CudaEngineWithRuntime> onnx2trt(
    std::string const& onnxModelPath,
    std::string name_suffix,  // model name后缀，用于判断传入的类型
    std::vector<std::vector<std::vector<int>>>&
        mins,  // multiple profiles - multiple inputs - multiDims
    std::vector<std::vector<std::vector<int>>>& maxs, std::string& engine_plan,
    OnnxParams& precision, const std::unordered_map<std::string, std::string>& int8_param,
    std::vector<float> means, std::vector<float> stds);

bool initPlugins();
std::shared_ptr<CudaEngineWithRuntime> loadCudaBackend(std::string const& trtModelPath,
                                                       const std::string& name_suffix,
                                                       std::string& engine_plan);
std::shared_ptr<CudaEngineWithRuntime> loadEngineFromBuffer(const std::string& engine_plan);
std::vector<std::vector<int>> infer_onnx_shape(std::string onnx_path);
}  // namespace ipipe