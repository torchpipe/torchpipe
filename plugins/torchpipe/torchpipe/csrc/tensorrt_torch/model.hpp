#pragma once
#include <hami/extension.hpp>
#include "NvInfer.h"
#include <NvInferRuntime.h>

namespace torchpipe {

// #if (NV_TENSORRT_MAJOR == 9 && NV_TENSORRT_MINOR < 3) || NV_TENSORRT_MAJOR <
// 9
#if (NV_TENSORRT_MAJOR == 10 && NV_TENSORRT_MINOR < 1) || NV_TENSORRT_MAJOR < 10
#error Only support TensorRT 10.1 and above
#endif

class LoadTensorrtEngine : public hami::Backend {
   public:
    ~LoadTensorrtEngine() {
        engine_.release();
        runtime_.release();
        allocator_.release();
    }

   private:
    void impl_init(const std::unordered_map<std::string, std::string>& config,
                   const hami::dict& dict_config) override;
    // void impl_forward(const std::vector<hami::dict>& input) override;

   private:
    std::unique_ptr<nvinfer1::IRuntime> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine> engine_;
    std::unique_ptr<nvinfer1::IGpuAllocator> allocator_;
};

class Onnx2Tensorrt : public hami::Backend {
   public:
    ~Onnx2Tensorrt() {
        engine_.release();
        runtime_.release();
        allocator_.release();
    }

   private:
    void impl_init(const std::unordered_map<std::string, std::string>& config,
                   const hami::dict& dict_config) override;
    // void impl_forward(const std::vector<hami::dict>& input) override;

   private:
    std::unique_ptr<nvinfer1::IRuntime> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine> engine_;
    std::unique_ptr<nvinfer1::IGpuAllocator> allocator_;
};

class ModelLoadder : public hami::Container {
   public:
    void post_init(const std::unordered_map<std::string, std::string>& config,
                   const hami::dict& dict_config) override;

   private:
    std::vector<size_t> set_init_order(size_t max_range) const override {
        return {};
    }
};

}  // namespace torchpipe