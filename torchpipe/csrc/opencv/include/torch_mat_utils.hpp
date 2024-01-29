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

#include <opencv2/core.hpp>

#include "Backend.hpp"
#include "dict.hpp"

namespace ipipe {
cv::Mat torchTensortoCVMat(at::Tensor tensor, bool deepcopy = false);
cv::Mat torchTensortoCVMatV2(at::Tensor tensor, bool deepcopy = false);

at::Tensor cvMat2TorchGPU(cv::Mat tensor, std::string data_format = "nchw");
at::Tensor cvMat2TorchCPU(cv::Mat tensor, bool deepcopy = false, std::string data_format = "nchw");

void imwrite(std::string name, at::Tensor tensor);

void imwrite(std::string name, any tensor);

at::Tensor get_tensor_from_any(any input);
cv::Mat get_mat_from_any(any input);

}  // namespace ipipe
