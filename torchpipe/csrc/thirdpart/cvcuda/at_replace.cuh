/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once
#include <ATen/ATen.h>
// #include "at_replace.cuh"
namespace ipipe_nvcv {


template <class T>  // base type
__host__ __device__ int32_t CalcNCHWImageStride(int rows, int cols, int channels) {
  return rows * cols * channels * sizeof(T);
}

template <class T>  // base type
__host__ __device__ int32_t CalcNCHWRowStride(int cols, int channels) {
  return cols * sizeof(T);
}

template <class T>  // base type
__host__ __device__ int32_t CalcNHWCImageStride(int rows, int cols, int channels) {
  return rows * cols * channels * sizeof(T);
}

template <class T>  // base type
__host__ __device__ int32_t CalcNHWCRowStride(int cols, int channels) {
  return cols * channels * sizeof(T);
}

// template <typename U>
// __host__ __device__ auto SaturateCast(U u) {
//   return u;
// }

__host__ __device__ unsigned char SaturateCastPillow(work_type in) {
  auto saturate_val = static_cast<intmax_t>(in) >> precision_bits;
  if (saturate_val < 0) {
    return 0;
  }
  if (saturate_val > UINT8_MAX) {
    return UINT8_MAX;
  }
  return static_cast<unsigned char>(saturate_val);

  // return u;
};
}  // namespace ipipe_nvcv
