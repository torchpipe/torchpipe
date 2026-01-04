// Copyright 2021-2025 NetEase.
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

namespace torchpipe {
using dict = omniback::dict;

#if 1
bool torch_not_use_default_stream(bool high_prio = false);
bool torch_not_use_default_stream(int device_id, bool high_prio = false);
bool torch_is_using_default_stream();
torch::Tensor to_current_device(torch::Tensor input);

#endif

torch::Tensor get_tensor_from_any(omniback::any input);
std::string print_tensor(
    const std::vector<torch::Tensor>& data,
    const std::string& tag = "");

bool is_any_cpu(omniback::any input);
bool is_cpu_tensor(torch::Tensor input);
static inline torch::TensorOptions get_tensor_option(c10::ScalarType dtype) {
  return torch::TensorOptions()
      .device(torch::kCUDA, -1)
      .dtype(dtype) // torch::kByte
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

enum class MemoryFormat { HWC, NCHW };
MemoryFormat guard_valid_memory_format(const torch::Tensor& data);
/**
 * @brief change hwc or 1chw to hwc and check c==1 or 3, 4
 *
 * @param input torch::Tensor(hwc or 1chw)
 * @return torch::Tensor(hwc)
 */
torch::Tensor img_hwc_guard(torch::Tensor in);

torch::Tensor tensor_permute(
    torch::Tensor input,
    const std::vector<int>& min_shape,
    const std::vector<int>& max_shape,
    bool& need_permute);

/**
 * @brief 保存 torch::Tensor 到文件。等价于 ``torch.save(name: str, tensor:
 * torch.Tensor)``
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

  return torch::empty(
      {size},
      torch::dtype(torch::kByte).device(torch::kCUDA),
      torch::MemoryFormat::Contiguous);
}

std::vector<torch::Tensor> get_tensors(
    omniback::dict input_dict,
    const std::string& key);

void copy2ptr(torch::Tensor input, char* ptr);
torch::Tensor try_quick_cat(std::vector<torch::Tensor> resized_inputs);

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
static inline int torch_malloc_async(
    void* ctx,
    void** p,
    size_t size,
    cudaStream_t stream) {
  (void)ctx; // Ignore the context pointer if not used
  if (size == 0) {
    *p = nullptr;
    return -1; // Error: Invalid size
  }

  *p = c10::cuda::CUDACachingAllocator::raw_alloc_with_stream(size, stream);
  if (*p == nullptr) {
    return -2; // Error: Memory allocation failed
  }

  return 0; // Success
}

// Async Memory Free with Error Handling
static inline int torch_free_async(
    void* ctx,
    void* p,
    size_t size,
    cudaStream_t stream) {
  (void)ctx; // Ignore the context pointer if not used
  if (p == nullptr) {
    return -3; // Error: Invalid pointer
  }

  c10::cuda::CUDACachingAllocator::raw_delete(p);
  return 0; // Success
}

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

std::string get_sm();

// torch::Tensor fix_and_cat_tensor(std::vector<torch::Tensor>& data,
//                                  const NetIOInfo& info);

void fix_tensor_shape(
    torch::Tensor& data,
    const NetIOInfo::Dims64 min,
    const NetIOInfo::Dims64& max);
void fix_tensor_type(torch::Tensor& input, NetIOInfo::DataType desired_type);
void fix_tensor_device(torch::Tensor& input, NetIOInfo::Device desired_device);
void fix_tensors(
    std::vector<torch::Tensor>& tensors,
    const std::shared_ptr<NetIOInfos>& infos);

torch::Device netinfo2torch_device(NetIOInfo::Device device);
c10::ScalarType netinfo2torch_type(NetIOInfo::DataType dtype);
void check_batched_inputs(
    const std::vector<torch::Tensor>& tensors,
    const std::vector<NetIOInfo>& infos);

bool match(NetIOInfo::Dims64& dst, const torch::Tensor& src);

c10::ScalarType netinfo2torch_type(NetIOInfo::DataType dtype);

float cuda_time();
// int StreamOrderedManagedTensorAllocator(
//     void* stream,
//     DLTensor* prototype,
//     DLManagedTensorVersioned** out,
//     void* error_ctx,
//     void (*SetError)(void* error_ctx, const char* kind, const char* message));
// device_count();
} // namespace torchpipe
