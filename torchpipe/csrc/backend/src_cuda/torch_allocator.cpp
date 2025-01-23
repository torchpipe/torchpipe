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

// todo vattention https://github.com/pytorch/pytorch/pull/86786
// https://github.com/pytorch/pytorch/issues/124807
void* TorchAllocator::allocate(uint64_t const size, uint64_t const alignment,
                               uint32_t const flags) noexcept {
  if (size == 0) return nullptr;
  // if (alignment == 0) {
  //   SPDLOG_ERROR("TorchAllocator::allocate failed(alignment={}!=0)", alignment);
  //   return nullptr;
  // }
  if (c10::cuda::getCurrentCUDAStream() == c10::cuda::getDefaultCUDAStream()) {
    SPDLOG_ERROR(
        "thread was binding to default cuda stream, which is not supported by TorchAllocator.");
    return nullptr;
  }

  try {
    torch::Tensor buf =
        torch::empty({static_cast<int64_t>(size)}, torch::dtype(torch::kByte).device(torch::kCUDA));
    void* ptr = buf.data_ptr();

    if (alignment && reinterpret_cast<uintptr_t>(ptr) % alignment != 0) {
      auto offset = (alignment - (reinterpret_cast<uintptr_t>(ptr) % alignment));
      buf = torch::Tensor();  // stream ordered reuse
      buf = torch::empty({static_cast<int64_t>(size + offset)},
                         torch::dtype(torch::kByte).device(torch::kCUDA));
      ptr = buf.data_ptr();
      ptr = (char*)ptr + offset;
    }
    {
      std::lock_guard<std::mutex> lock(mutex_);

      data_.insert({ptr, buf});
    }

    c10::cuda::getCurrentCUDAStream().synchronize();
    return ptr;
  } catch (const std::exception& e) {
    SPDLOG_ERROR("TorchAllocator::allocate failed(size={}MB): {}", size / 1024.0 / 1024.0,
                 e.what());
    return nullptr;
  }

  return nullptr;
}

bool TorchAllocator::deallocate(void* const memory) noexcept {
  if (memory == nullptr) return true;
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = data_.find(memory);
  if (it != data_.end()) {
    data_.erase(it);
    return true;
  }
  SPDLOG_DEBUG("TorchAllocator::deallocate failed(memory={})", memory);
  return false;
}

#if NV_TENSORRT_MAJOR >= 10
void* TorchAllocator::allocateAsync(uint64_t const size, uint64_t const alignment,
                                    uint32_t const flags, cudaStream_t stream) noexcept {
  // if (alignment == 0) {
  //   SPDLOG_ERROR("TorchAllocator::allocateAsync failed(alignment={}!=0)", alignment);
  //   return nullptr;
  // }
  if (size == 0) return nullptr;

  // todo: CUDAStreamGuard, getStreamFromExternal
  if (stream == nullptr || c10::cuda::getCurrentCUDAStream() != stream) {
    SPDLOG_ERROR("TorchAllocator was not using current cuda stream, which is not supported.");
    return nullptr;
  }
  SPDLOG_DEBUG("TorchAllocator::allocateAsync(size={}, alignment={}, flags={})", size, alignment,
               flags);

  try {
    torch::Tensor buf =
        torch::empty({static_cast<int64_t>(size)}, torch::dtype(torch::kByte).device(torch::kCUDA));
    void* ptr = buf.data_ptr();
    // void* aligned_ptr = ptr;
    // if (alignment && reinterpret_cast<uintptr_t>(ptr) % alignment != 0) {
    //   // 计算对齐后的指针
    //   aligned_ptr = (char*)ptr + (alignment - reinterpret_cast<uintptr_t>(ptr) % alignment);
    //   IPIPE_ASSERT(reinterpret_cast<uintptr_t>(aligned_ptr) % alignment == 0);
    // }

    if (alignment && reinterpret_cast<uintptr_t>(ptr) % alignment != 0) {
      auto offset = (alignment - (reinterpret_cast<uintptr_t>(ptr) % alignment));
      buf = torch::Tensor();  // stream ordered reuse
      buf = torch::empty({static_cast<int64_t>(size + offset)},
                         torch::dtype(torch::kByte).device(torch::kCUDA));
      ptr = buf.data_ptr();
      ptr = (char*)ptr + offset;
    }

    std::lock_guard<std::mutex> lock(mutex_);

    data_.insert({ptr, buf});
    return ptr;
  } catch (const std::exception& e) {
    SPDLOG_ERROR("TorchAllocator::allocateAsync failed(size={}): {}", size, e.what());
    return nullptr;
  }

  return nullptr;
}

bool TorchAllocator::deallocateAsync(void* const memory, cudaStream_t) noexcept {
  SPDLOG_DEBUG("TorchAllocator::deallocateAsync(memory={})", memory);
  if (memory == nullptr) return true;
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = data_.find(memory);
  if (it != data_.end()) {
    data_.erase(it);
    return true;
  }
  SPDLOG_ERROR("TorchAllocator::deallocateAsync failed(memory={})", memory);
  return false;
}  // override;

void* TorchAsyncAllocator::allocateAsync(uint64_t const size, uint64_t const alignment,
                                         nvinfer1::AllocatorFlags const flags,
                                         cudaStream_t stream) noexcept {
  // if (alignment == 0) {
  //   SPDLOG_ERROR("TorchAllocator::allocateAsync failed(alignment={}!=0)", alignment);
  //   return nullptr;
  // }
  if (size == 0) return nullptr;
  if (stream == nullptr) {
    return c10::cuda::CUDACachingAllocator::raw_alloc_with_stream(size, nullptr);
    // todo aligned?
  }

  if (c10::cuda::getCurrentCUDAStream() != stream) {
    SPDLOG_ERROR("TorchAsyncAllocator was not using current cuda stream, which is not supported.");
    return nullptr;
  }

  try {
    torch::Tensor buf =
        torch::empty({static_cast<int64_t>(size)}, torch::dtype(torch::kByte).device(torch::kCUDA));
    void* ptr = buf.data_ptr();

    if (alignment && reinterpret_cast<uintptr_t>(ptr) % alignment != 0) {
      auto offset = (alignment - (reinterpret_cast<uintptr_t>(ptr) % alignment));
      buf = torch::Tensor();  // stream ordered reuse
      buf = torch::empty({static_cast<int64_t>(size + offset)},
                         torch::dtype(torch::kByte).device(torch::kCUDA));
      ptr = buf.data_ptr();
      ptr = (char*)ptr + offset;
    }

    std::lock_guard<std::mutex> lock(mutex_);

    data_.insert({ptr, buf});
    return ptr;
  } catch (const std::exception& e) {
    SPDLOG_ERROR("TorchAllocator::allocateAsync failed(size={}): {}", size, e.what());
    return nullptr;
  }

  return nullptr;
}

void* TorchAsyncOutAllocator::reallocateOutputAsync(char const* tensorName, void* currentMemory,
                                                    uint64_t size, uint64_t alignment,
                                                    cudaStream_t stream) noexcept {
  if (c10::cuda::getCurrentCUDAStream() != stream) {
    SPDLOG_ERROR("TorchAsyncAllocator was not using current cuda stream, which is not supported.");
    return nullptr;
  }

  size = std::max(size, static_cast<uint64_t>(1));
  if (size > allocated_size_) {
    try {
      torch::Tensor buf = torch::empty({static_cast<int64_t>(size)},
                                       torch::dtype(torch::kByte).device(torch::kCUDA));
      void* ptr = buf.data_ptr();

      if (alignment && reinterpret_cast<uintptr_t>(ptr) % alignment != 0) {
        auto offset = (alignment - (reinterpret_cast<uintptr_t>(ptr) % alignment));
        buf = torch::Tensor();  // stream ordered reuse
        buf = torch::empty({static_cast<int64_t>(size + offset)},
                           torch::dtype(torch::kByte).device(torch::kCUDA));
        ptr = buf.data_ptr();
        ptr = (char*)ptr + offset;
      }

      // std::lock_guard<std::mutex> lock(mutex_);
      // data_.insert({ptr, buf});
      allocated_size_ = size;
      ptr_ = ptr;

    } catch (const std::exception& e) {
      SPDLOG_ERROR("TorchAllocator::allocateAsync failed(size={}): {}", size, e.what());
      return nullptr;
    }
  }
  return ptr_;
}

void TorchAsyncOutAllocator::notifyShape(char const* tensorName,
                                         nvinfer1::Dims const& dims) noexcept {}

bool TorchAsyncAllocator::deallocateAsync(void* const memory, cudaStream_t stream) noexcept {
  // if (memory == nullptr) return true;

  if (stream == nullptr) {
    c10::cuda::CUDACachingAllocator::raw_delete(memory);
    return true;
  }
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = data_.find(memory);
  if (it != data_.end()) {
    data_.erase(it);
    return true;
  }
  // https://pytorch.org/docs/stable/generated/torch.Tensor.record_stream.html#torch.Tensor.record_stream
  // no need to record stream
  return false;
}  // override;
#endif

}  // namespace ipipe

#endif
#endif