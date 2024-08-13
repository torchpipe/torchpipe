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

#include "NvInferRuntimeCommon.h"
#include <unordered_map>
#include <torch/torch.h>
#include <mutex>

#include <c10/cuda/CUDACachingAllocator.h>

namespace ipipe {
class TorchAllocator : public nvinfer1::IGpuAllocator {
 public:
  TorchAllocator() = default;

  void* allocate(uint64_t size, uint64_t alignment, uint32_t flags) noexcept override;
#if NV_TENSORRT_MAJOR < 10
  void free(void* const memory) noexcept override;
#endif
  bool deallocate(void* const memory) noexcept;  // override;

 private:
  std::unordered_map<void*, torch::Tensor> data_;
  std::mutex mutex_;
};

int static inline dev_malloc(void** p, size_t s) {
  *p = c10::cuda::CUDACachingAllocator::raw_alloc(s);
  return 0;
}

int static inline dev_free(void* p) {
  assert(p != nullptr);
  c10::cuda::CUDACachingAllocator::raw_delete(p);
  return 0;
}
}  // namespace ipipe
