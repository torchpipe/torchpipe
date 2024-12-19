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

#include "NvInferRuntime.h"
#include <unordered_map>
#include <torch/torch.h>
#include <mutex>

#include <c10/cuda/CUDACachingAllocator.h>

// GLake: Optimizing GPU memory management & IO transmission
// vattention
// https://github.com/intelligent-machine-learning/glake
// https://github.com/intelligent-machine-learning/glake/issues/12
namespace ipipe {
class TorchAllocator : public nvinfer1::IGpuAllocator {
 public:
  TorchAllocator() = default;

  void* allocate(uint64_t const size, uint64_t const alignment,
                 uint32_t const flags) noexcept override;

#if NV_TENSORRT_MAJOR >= 10
  void* allocateAsync(uint64_t const size, uint64_t const alignment, uint32_t const flags,
                      cudaStream_t) noexcept override;
  bool deallocateAsync(void* const memory, cudaStream_t) noexcept override;  // override;

#endif

#if NV_TENSORRT_MAJOR < 10
  void free(void* const memory) noexcept override { deallocate(memory); }
#endif
  bool deallocate(void* const memory) noexcept;  // override;

 private:
  std::unordered_map<void*, torch::Tensor> data_;
  std::mutex mutex_;
};

#if NV_TENSORRT_MAJOR >= 10
class TorchAsyncAllocator : public nvinfer1::IGpuAsyncAllocator {
 public:
  void* allocateAsync(uint64_t const size, uint64_t const alignment,
                      nvinfer1::AllocatorFlags const flags, cudaStream_t) noexcept override;
  bool deallocateAsync(void* const memory, cudaStream_t) noexcept override;  // override;

 private:
  std::unordered_map<void*, torch::Tensor> data_;
  std::mutex mutex_;
};

class TorchAsyncOutAllocator : public nvinfer1::IOutputAllocator {
  void* reallocateOutputAsync(char const* tensorName, void* currentMemory, uint64_t size,
                              uint64_t alignment, cudaStream_t /*stream*/) noexcept override;

  /// Notify the shape of the tensor.
  void notifyShape(char const* tensorName, nvinfer1::Dims const& dims) noexcept override;

 private:
  std::unordered_map<void*, torch::Tensor> data_;
  std::mutex mutex_;
  size_t allocated_size_{0};
  void* ptr_{nullptr};
};

#endif

int static inline dev_malloc(void** p, size_t s) {
  *p = c10::cuda::CUDACachingAllocator::raw_alloc_with_stream(s, nullptr);
  // c10::cuda::getCurrentCUDAStream().synchronize();
  return 0;
}

int static inline dev_free(void* p) {
  assert(p != nullptr);
  c10::cuda::CUDACachingAllocator::raw_delete(p);
  return 0;
}

int static inline dev_malloc_async(void* ctx, void** p, size_t size, cudaStream_t stream) {
  *p = c10::cuda::CUDACachingAllocator::raw_alloc_with_stream(size, stream);
  return 0;
}

int static inline dev_free_async(void* ctx, void* p, size_t size, cudaStream_t stream) {
  assert(p != nullptr);
  c10::cuda::CUDACachingAllocator::raw_delete(p);
  return 0;
}
}  // namespace ipipe
