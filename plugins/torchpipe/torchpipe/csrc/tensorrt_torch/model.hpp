#pragma once
#include <NvInferRuntime.h>
#include <omniback/extension.hpp>
#include "NvInfer.h"

namespace torchpipe {

// #if (NV_TENSORRT_MAJOR == 9 && NV_TENSORRT_MINOR < 3) || NV_TENSORRT_MAJOR <
// 9
// #if (NV_TENSORRT_MAJOR == 10 && NV_TENSORRT_MINOR < 1) || NV_TENSORRT_MAJOR <
// 10 #error Only support TensorRT 10.1 and above
#if (NV_TENSORRT_MAJOR == 8 && NV_TENSORRT_MINOR < 5) || NV_TENSORRT_MAJOR < 8
#error Only support TensorRT >= 8.5
#endif

class LoadTensorrtEngine : public omniback::Backend {
 public:
  ~LoadTensorrtEngine() {
    engine_.release();
    runtime_.release();
    allocator_.release();
  }

 private:
  void impl_init(
      const std::unordered_map<std::string, std::string>& config,
      const omniback::dict& kwargs) override;
  // void impl_forward(const std::vector<omniback::dict>& input) override;

 private:
  std::unique_ptr<nvinfer1::IRuntime> runtime_;
  std::unique_ptr<nvinfer1::ICudaEngine> engine_;
  std::unique_ptr<nvinfer1::IGpuAllocator> allocator_;
};

class Onnx2Tensorrt : public omniback::Backend {
 public:
  ~Onnx2Tensorrt() {
    engine_.release();
    runtime_.release();
    allocator_.release();
  }

 private:
  void impl_init(
      const std::unordered_map<std::string, std::string>& config,
      const omniback::dict& kwargs) override;
  // void impl_forward(const std::vector<omniback::dict>& input) override;

 private:
  std::unique_ptr<nvinfer1::IRuntime> runtime_;
  std::unique_ptr<nvinfer1::ICudaEngine> engine_;
  std::unique_ptr<nvinfer1::IGpuAllocator> allocator_;
};

class ModelLoadder : public omniback::Container {
 public:
  void post_init(
      const std::unordered_map<std::string, std::string>& config,
      const omniback::dict& kwargs) override;

 private:
  std::vector<uint32_t> set_init_order(uint32_t max_range) const override {
    return {};
  }
};

} // namespace torchpipe
