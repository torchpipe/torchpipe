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
#include <torch/torch.h>

namespace ipipe_nvcv {

using TensorDataAccessStridedImagePlanar = torch::Tensor;
typedef enum {
  NVCV_INTERP_NEAREST = 0,
  NVCV_INTERP_LINEAR = 1,
  NVCV_INTERP_CUBIC = 2,
  NVCV_INTERP_AREA = 3,
  NVCV_INTERP_MAX = 7,
  NVCV_WARP_INVERSE_MAP = 16
} NVCVInterpolationType;

// @brief Flag to choose the border mode to be used
typedef enum {
  NVCV_BORDER_CONSTANT = 0,
  NVCV_BORDER_REPLICATE = 1,
  NVCV_BORDER_REFLECT = 2,
  NVCV_BORDER_WRAP = 3,
  NVCV_BORDER_REFLECT101 = 4,
} NVCVBorderType;

using DataType = torch::ScalarType;
using DataShape = torch::IntArrayRef;
inline size_t DataSize(DataType data_type) {
  size_t size = 0;
  switch (data_type) {
    case torch::kByte:
    case torch::kChar:
      size = 1;
      break;
    case torch::kShort:
    case torch::kHalf:
      size = 2;
      break;
    case torch::kInt:
    case torch::kFloat:
      size = 4;
      break;
    case torch::kDouble:
      size = 8;
    case torch::kLong:
      size = 8;

    default:
      throw std::runtime_error("it is not allowed that sizeof(data_type) > 4");
      break;
  }
  return size;
}

inline int divUp(int a, int b) {
  assert(b > 0);
  return std::ceil((float)a / b);
};

typedef struct {
  int32_t x;       //!< x coordinate of the top-left corner
  int32_t y;       //!< y coordinate of the top-left corner
  int32_t width;   //!< width of the rectangle
  int32_t height;  //!< height of the rectangle
} NVCVRectI;

#define get_batch_idx() (blockIdx.z)
#define get_lid() (threadIdx.y * blockDim.x + threadIdx.x)

#ifndef checkKernelErrors
#define checkKernelErrors(expr)                                                         \
  do {                                                                                  \
    expr;                                                                               \
                                                                                        \
    cudaError_t __err = cudaGetLastError();                                             \
    if (__err != cudaSuccess) {                                                         \
      printf("Line %d: '%s' failed: %s\n", __LINE__, #expr, cudaGetErrorString(__err)); \
      throw std::runtime_error(cudaGetErrorString(__err));                              \
    }                                                                                   \
  } while (0)
#endif

}  // namespace ipipe_nvcv