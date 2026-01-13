
#include "tensorrt_torch/allocator.hpp"
#include <omniback/extension.hpp>
#include "c10/cuda/CUDACachingAllocator.h"
#include "c10/cuda/CUDAStream.h"

namespace torchpipe {

#if (NV_TENSORRT_MAJOR >= 10)
void* TorchAsyncAllocator::allocateAsync(
    uint64_t const size,
    uint64_t const alignment,
    nvinfer1::AllocatorFlags const flags,
    cudaStream_t stream) noexcept {
  // if (alignment == 0) {
  //   SPDLOG_ERROR("TorchAllocator::allocateAsync failed(alignment={}!=0)",
  //   alignment); return nullptr;
  // }
  [[maybe_unused]] static const auto __tmp__allocator_init_ = []() {
    c10::cuda::CUDACachingAllocator::init(c10::cuda::device_count());
    return 0;
  }();
  if (size == 0)
    return nullptr;
  if (stream == nullptr || c10::cuda::getCurrentCUDAStream() != stream) {
    void* ptr = nullptr;
    SPDLOG_INFO("trt: cudaMallocAsync size={}", size);
    cudaError_t err = cudaMallocAsync(&ptr, size, stream);
    if (err != cudaSuccess) {
      SPDLOG_ERROR("cudaMallocAsync failed  {}", cudaGetErrorString(err));
      return nullptr;
    }
    return ptr;
  }

  try {
    torch::Tensor buf = torch::empty(
        {static_cast<int64_t>(size)},
        torch::dtype(torch::kByte).device(torch::kCUDA));
    void* ptr = buf.data_ptr();

    if (alignment && reinterpret_cast<uintptr_t>(ptr) % alignment != 0) {
      auto offset =
          (alignment - (reinterpret_cast<uintptr_t>(ptr) % alignment));
      buf = torch::Tensor(); // stream ordered reuse.release previous
                             //  memory
      buf = torch::empty(
          {static_cast<int64_t>(size + offset)},
          torch::dtype(torch::kByte).device(torch::kCUDA));
      ptr = buf.data_ptr();
      ptr = (char*)ptr + offset;
    }

    std::lock_guard<std::mutex> lock(mutex_);

    data_.insert({ptr, buf});

    return ptr;
  } catch (const std::exception& e) {
    SPDLOG_ERROR(
        "TorchAllocator::allocateAsync failed(size={}): {}", size, e.what());
    return nullptr;
  }

  return nullptr;
}

bool TorchAsyncAllocator::deallocateAsync(
    void* const memory,
    cudaStream_t stream) noexcept {
  bool is_torch_stream = false;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    is_torch_stream = data_.find(memory) != data_.end();
  }
  if (!is_torch_stream) {
    cudaError_t err = cudaFreeAsync(memory, stream);
    if (err != cudaSuccess) {
      SPDLOG_ERROR("cudaFreeAsync failed: {}", cudaGetErrorString(err));
      return false;
    }
    return true;
  }

  std::lock_guard<std::mutex> lock(mutex_);
  data_.erase(memory);
  return true;

} // override;
#endif
} // namespace torchpipe