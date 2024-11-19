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

namespace ipipe {

#ifdef WITH_CUDA
bool torch_not_use_default_stream(bool high_prio = false);
bool torch_not_use_default_stream(int device_id, bool high_prio = false);
bool torch_is_using_default_stream();
torch::Tensor to_current_device(torch::Tensor input);

#endif

torch::Tensor get_tensor_from_any(any input);

bool is_any_cpu(any input);
bool is_cpu_tensor(torch::Tensor input);
static inline torch::TensorOptions get_tensor_option(c10::ScalarType dtype) {
  return torch::TensorOptions()
      .device(torch::kCUDA, -1)
      .dtype(dtype)  // torch::kByte
      .layout(torch::kStrided)
      .requires_grad(false);
}

torch::Tensor switch_device(torch::Tensor input);
torch::Tensor async2cpu(torch::Tensor input);

// torch::Tensor tensor2nchw(torch::Tensor, int& n, int& c, int& h, int& w);
torch::Tensor tensor2nchw(torch::Tensor in);
/**
 * @brief change hwc or 1chw to 1chw and check c==1 or 3, 4
 *
 * @param input torch::Tensor(hwc or 1chw)
 * @return torch::Tensor(1chw)
 */
torch::Tensor img_1chw_guard(torch::Tensor input);
torch::Tensor img_nchw_guard(torch::Tensor input);
torch::Tensor img_1hwc_guard(torch::Tensor input);

bool is_hwc(torch::Tensor in);
bool is_1chw(torch::Tensor in);
bool is_nchw(torch::Tensor in);

/**
 * @brief change hwc or 1chw to hwc and check c==1 or 3, 4
 *
 * @param input torch::Tensor(hwc or 1chw)
 * @return torch::Tensor(hwc)
 */
torch::Tensor img_hwc_guard(torch::Tensor in);

torch::Tensor tensor_permute(torch::Tensor input, const std::vector<int>& min_shape,
                             const std::vector<int>& max_shape, bool& need_permute);

/**
 * @brief 保存 torch::Tensor 到文件。等价于 ``torch.save(name: str, tensor: torch.Tensor)``
 *
 * @param name 文件路径
 * @param input 输入的tensor
 */
void save(std::string name, torch::Tensor input);

torch::Tensor load_tensor(std::string name);
bool is_contiguous_wrt_hwc(torch::Tensor in);
bool is_contiguous_wrt_nchw(torch::Tensor in);

static inline torch::Tensor torch_allocate(int64_t size) {
  // auto options = torch::TensorOptions()
  //                    .device(torch::kCUDA, -1)
  //                    .dtype(torch::kByte)
  //                    .layout(torch::kStrided)
  //                    .requires_grad(false);
  // return torch::empty({size}, options, torch::MemoryFormat::Contiguous);

  return torch::empty({size}, torch::dtype(torch::kByte).device(torch::kCUDA),
                      torch::MemoryFormat::Contiguous);
}

std::vector<torch::Tensor> get_tensors(dict input_dict, const std::string& key);

void copy2ptr(torch::Tensor input, char* ptr);
torch::Tensor try_quick_cat(std::vector<torch::Tensor> resized_inputs);
}  // namespace ipipe
