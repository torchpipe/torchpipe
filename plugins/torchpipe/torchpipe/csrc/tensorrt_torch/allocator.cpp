
#include "tensorrt_torch/allocator.hpp"
#include <hami/extension.hpp>

namespace torchpipe {

void* TorchAsyncAllocator::allocateAsync(uint64_t const size,
                                         uint64_t const alignment,
                                         nvinfer1::AllocatorFlags const flags,
                                         cudaStream_t stream) noexcept {
    // if (alignment == 0) {
    //   SPDLOG_ERROR("TorchAllocator::allocateAsync failed(alignment={}!=0)",
    //   alignment); return nullptr;
    // }
    if (size == 0) return nullptr;
    if (stream == nullptr) {
        return c10::cuda::CUDACachingAllocator::raw_alloc_with_stream(size,
                                                                      nullptr);
        // todo aligned?
    }

    if (c10::cuda::getCurrentCUDAStream() != stream) {
        SPDLOG_ERROR(
            "TorchAsyncAllocator was not using current cuda stream, which is "
            "not supported.");
        return nullptr;
    }

    try {
        torch::Tensor buf =
            torch::empty({static_cast<int64_t>(size)},
                         torch::dtype(torch::kByte).device(torch::kCUDA));
        void* ptr = buf.data_ptr();

        if (alignment && reinterpret_cast<uintptr_t>(ptr) % alignment != 0) {
            auto offset =
                (alignment - (reinterpret_cast<uintptr_t>(ptr) % alignment));
            buf = torch::Tensor();  // stream ordered reuse.release previous
                                    //  memory
            buf = torch::empty({static_cast<int64_t>(size + offset)},
                               torch::dtype(torch::kByte).device(torch::kCUDA));
            ptr = buf.data_ptr();
            ptr = (char*)ptr + offset;
        }

        std::lock_guard<std::mutex> lock(mutex_);

        data_.insert({ptr, buf});
        return ptr;
    } catch (const std::exception& e) {
        SPDLOG_ERROR("TorchAllocator::allocateAsync failed(size={}): {}", size,
                     e.what());
        return nullptr;
    }

    return nullptr;
}

bool TorchAsyncAllocator::deallocateAsync(void* const memory,
                                          cudaStream_t stream) noexcept {
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
}  // namespace torchpipe