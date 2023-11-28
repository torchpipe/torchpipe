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

#include "Backend.hpp"
#include "dict.hpp"

namespace ipipe {

bool torch_not_use_default_stream(bool high_prio = false);
bool torch_not_use_default_stream(int device_id, bool high_prio = false);
bool torch_is_using_default_stream();

at::Tensor get_tensor_from_any(any input);

bool is_any_cpu(any input);
bool is_cpu_tensor(at::Tensor input);
static inline at::TensorOptions get_tensor_option(c10::ScalarType dtype) {
  return at::TensorOptions()
      .device(at::kCUDA, -1)
      .dtype(dtype)  // at::kByte
      .layout(at::kStrided)
      .requires_grad(false);
}

at::Tensor switch_device(at::Tensor input);
at::Tensor to_current_device(at::Tensor input);
at::Tensor async2cpu(at::Tensor input);

at::Tensor tensor2nchw(at::Tensor, int& n, int& c, int& h, int& w);

/**
 * @brief change hwc or 1chw to 1chw and check c==1 or 3, 4
 *
 * @param input at::Tensor(hwc or 1chw)
 * @return at::Tensor(1chw)
 */
at::Tensor img_1chw_guard(at::Tensor input);
at::Tensor img_1hwc_guard(at::Tensor input);

bool is_hwc(at::Tensor in);
bool is_1chw(at::Tensor in);
bool is_nchw(at::Tensor in);

/**
 * @brief change hwc or 1chw to hwc and check c==1 or 3, 4
 *
 * @param input at::Tensor(hwc or 1chw)
 * @return at::Tensor(hwc)
 */
at::Tensor img_hwc_guard(at::Tensor in);

at::Tensor tensor_permute(at::Tensor input, const std::vector<int>& min_shape,
                          const std::vector<int>& max_shape, bool& need_permute);

/**
 * @brief 保存 at::Tensor 到文件。等价于 ``torch.save(name: str, tensor: torch.Tensor)``
 *
 * @param name 文件路径
 * @param input 输入的tensor
 */
void save(std::string name, at::Tensor input);

at::Tensor load_tensor(std::string name);

static inline at::Tensor torch_allocate(int64_t size) {
  // auto options = at::TensorOptions()
  //                    .device(at::kCUDA, -1)
  //                    .dtype(at::kByte)
  //                    .layout(at::kStrided)
  //                    .requires_grad(false);
  // return at::empty({size}, options, at::MemoryFormat::Contiguous);

  return at::empty({size}, at::dtype(at::kByte).device(at::kCUDA), at::MemoryFormat::Contiguous);
}

at::Tensor try_quick_cat(std::vector<at::Tensor> resized_inputs);
}  // namespace ipipe
