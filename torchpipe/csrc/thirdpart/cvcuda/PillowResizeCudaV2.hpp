/* Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
 * SPDX-License-Identifier: Apache-2.0
 *
 * Copyright (C) 2021-2022, Bytedance Inc. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once
#include "at_replace.hpp"
#include "c10/cuda/CUDAStream.h"
namespace ipipe_nvcv {

using work_type = double;
static constexpr float bilinear_filter_support = 1.;
static constexpr unsigned int precision_bits = 32 - 8 - 2;
static constexpr work_type const_init_buffer = 1 << (precision_bits - 1);

class PillowResizeCudaV2 {
 public:
  PillowResizeCudaV2(DataShape max_input_shape, std::size_t resize_h, std::size_t resize_w,
                     DataType max_data_type)
      : resize_h_(resize_h), resize_w_(resize_w) {
    int max_support = 1;  // 3
    size_t size =
        std::ceil(resize_h * (((1.0 * max_input_shape[2] / resize_h + 1) * max_support * 2 + 1) *
                                  sizeof(work_type) +
                              2 * sizeof(int)) +
                  resize_w * (((1.0 * max_input_shape[3] / resize_w + 1) * max_support * 2 + 1) *
                                  sizeof(work_type) +
                              2 * sizeof(int))) +
        max_input_shape[0] * max_input_shape[1] * max_input_shape[2] * resize_w *
            DataSize(max_data_type);
    auto __err = (cudaMalloc(&gpu_workspace_, size));
    if (__err != cudaSuccess) {
      throw std::runtime_error(cudaGetErrorString(__err));
    }
  }

  ~PillowResizeCudaV2() { cudaFree(gpu_workspace_); }
  at::Tensor forward(at::Tensor data) {
    // data = data.permute({0, 2, 3, 1}).contiguous();
    auto options = at::TensorOptions()
                       .device(at::kCUDA, -1)
                       .dtype(data.dtype())
                       .layout(at::kStrided)
                       .requires_grad(false);
    auto image_tensor =
        at::empty({(signed long)1, (signed long)resize_h_, (signed long)resize_w_, (signed long)3},
                  options, at::MemoryFormat::Contiguous);
    if (!data.is_contiguous()) {
      data = data.contiguous();
    }
    forward_impl(data, image_tensor);
    return image_tensor;
  }

 private:
  void forward_impl(at::Tensor data, at::Tensor out);
  void* gpu_workspace_;
  std::size_t resize_h_;
  std::size_t resize_w_;
};

}  // namespace ipipe_nvcv
