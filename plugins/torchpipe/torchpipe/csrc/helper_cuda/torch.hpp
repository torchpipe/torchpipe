// Copyright 2021-2026 NetEase.
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
// #include <omniback/extension.hpp>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAStream.h>
#include <omniback/core/dict.hpp>
#include <torch/torch.h>
#include "helper/net_info.hpp"
#include "omniback/ffi/type_traits.h"
#include <omniback/addons/torch/type_traits.h>
#include "helper/mat.hpp"

namespace torchpipe {
using dict = om::dict;


bool torch_not_use_default_stream(bool high_prio = false);
bool torch_not_use_default_stream(int device_id, bool high_prio = false);
bool torch_is_using_default_stream();
torch::Tensor to_current_device(torch::Tensor input);
float cuda_time();


int static inline torch_malloc(void** p, size_t s) {
  *p = c10::cuda::CUDACachingAllocator::raw_alloc_with_stream(s, nullptr);
  // c10::cuda::getCurrentCUDAStream().synchronize();
  return 0;
}

int static inline torch_free(void* p) {
  assert(p != nullptr);
  c10::cuda::CUDACachingAllocator::raw_delete(p);
  return 0;
}

// Async Memory Allocation with Error Handling
int torch_malloc_async(
    void* ctx,
    void** p,
    size_t size,
    cudaStream_t stream);

// Async Memory Free with Error Handling
int torch_free_async(
    void* ctx,
    void* p,
    size_t size,
    cudaStream_t stream);

// Pinned Memory Allocator Using PyTorch
int static inline torch_pinned_malloc_async(
    void* ctx,
    void** p,
    size_t size,
    cudaStream_t stream) {
  // Check for zero allocation
  if (size == 0) {
    *p = nullptr;
    return 0;
  }

  // Allocate pinned memory using CUDA runtime
  cudaError_t cuda_err = cudaHostAlloc(p, size, cudaHostAllocDefault);
  if (cuda_err != cudaSuccess) {
    // Handle error (e.g., return negative error code)
    return -1; // Error code for failure
  }

  return 0;
}

int static inline torch_pinned_free_async(
    void* ctx,
    void* p,
    size_t size,
    cudaStream_t stream) {
  assert(p != nullptr); // Ensure pointer is valid

  // Free pinned memory using CUDA runtime
  cudaError_t cuda_err = cudaFreeHost(p);
  if (cuda_err != cudaSuccess) {
    return -1; // Error code for failure
  }

  return 0;
}

} // namespace torchpipe
