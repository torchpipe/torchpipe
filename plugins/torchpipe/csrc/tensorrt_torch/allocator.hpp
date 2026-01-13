#include <unordered_map>
#include <mutex>
#include "NvInfer.h"
#include <NvInferRuntime.h>

#include <torch/torch.h>

namespace torchpipe {

#if (NV_TENSORRT_MAJOR >= 10)
class TorchAsyncAllocator : public nvinfer1::IGpuAsyncAllocator {
 public:
  void* allocateAsync(
      uint64_t const size,
      uint64_t const alignment,
      nvinfer1::AllocatorFlags const flags,
      cudaStream_t) noexcept override;
  bool deallocateAsync(
      void* const memory,
      cudaStream_t) noexcept override; // override;

 private:
  std::unordered_map<void*, torch::Tensor> data_;
  std::mutex mutex_;
};
#endif
} // namespace torchpipe