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

#include "NvInferRuntimeCommon.h"
#include <unordered_map>
#include <ATen/ATen.h>
#include <mutex>
namespace ipipe {
class TorchAllocator : public nvinfer1::IGpuAllocator {
 public:
  TorchAllocator() = default;

  void* allocate(uint64_t size, uint64_t alignment, uint32_t flags) noexcept override;

  void free(void* const memory) noexcept override;
  bool deallocate(void* const memory) noexcept override;

 private:
  std::unordered_map<void*, at::Tensor> data_;
  std::mutex mutex_;
};
}  // namespace ipipe
