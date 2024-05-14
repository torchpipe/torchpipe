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
#ifdef WITH_TENSORRT
#include "NvInferVersion.h"
#if (NV_TENSORRT_MAJOR >= 9 || (NV_TENSORRT_MAJOR >= 8 && NV_TENSORRT_MINOR >= 6))
#include <torch/torch.h>

#include "Backend.hpp"
#include "dict.hpp"
#include "NvInferRuntimeCommon.h"
#include "torch_allocator.hpp"
#include "base_logging.hpp"
namespace ipipe {

void* TorchAllocator::allocate(uint64_t size, uint64_t alignment, uint32_t flags) noexcept {
  if (alignment == 0) {
    return nullptr;
  }
  if (size % alignment != 0) {
    size = (size / alignment + 1) * alignment;
  }
  try {
    torch::Tensor buf =
        torch::empty({static_cast<int64_t>(size)}, torch::dtype(torch::kByte).device(torch::kCUDA));
    std::lock_guard<std::mutex> lock(mutex_);
    void* ptr = buf.data_ptr();
    data_.insert({ptr, buf});
    return ptr;
  } catch (const std::exception& e) {
    SPDLOG_ERROR("TorchAllocator::allocate failed(size={}): {}", size, e.what());
    return nullptr;
  }

  return nullptr;
}

#if NV_TENSORRT_MAJOR < 10
void TorchAllocator::free(void* const memory) noexcept { deallocate(memory); }
#endif

bool TorchAllocator::deallocate(void* const memory) noexcept {
  if (memory == nullptr) return true;
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = data_.find(memory);
  if (it != data_.end()) {
    data_.erase(it);
    return true;
  }
  return false;
}

}  // namespace ipipe

#endif
#endif