#pragma once
#include <hami/extension.hpp>
#include "NvInfer.h"
#include <NvInferRuntime.h>

namespace torchpipe {

class LoadTensorrtEngine : public hami::Backend {
   public:
    void init(const std::unordered_map<std::string, std::string>& config,
              const hami::dict& dict_config) override;
    // void forward(const std::vector<hami::dict>& input) override;

   private:
    std::unique_ptr<nvinfer1::IRuntime> runtime_;
    std::shared_ptr<nvinfer1::ICudaEngine> engine_;
    std::unique_ptr<nvinfer1::IGpuAllocator> allocator_;
};

class Onnx2Tensorrt : public hami::Backend {
   public:
    void init(const std::unordered_map<std::string, std::string>& config,
              const hami::dict& dict_config) override;
    // void forward(const std::vector<hami::dict>& input) override;

   private:
    std::unique_ptr<nvinfer1::IRuntime> runtime_;
    std::shared_ptr<nvinfer1::ICudaEngine> engine_;
    std::unique_ptr<nvinfer1::IGpuAllocator> allocator_;
};

class ModelLoadder : public hami::Container {
   public:
    void post_init(const std::unordered_map<std::string, std::string>& config,
                   const hami::dict& dict_config) override;
    // void forward(const std::vector<hami::dict>& input) override;

   private:
    std::vector<size_t> set_init_order(size_t max_range) const override {
        return {};
    }
    // std::unique_ptr<nvinfer1::IRuntime> runtime_;
    // std::shared_ptr<nvinfer1::ICudaEngine> engine_;
    // std::unique_ptr<nvinfer1::IGpuAllocator> allocator_;
};

// class TensorrtContext : public hami::Backend {
//    public:
//     void init(const std::unordered_map<std::string, std::string>& config,
//               const hami::dict& dict_config) override;
//     // void forward(const std::vector<hami::dict>& input) override;

//    private:
//     std::shared_ptr<nvinfer1::ICudaEngine> engine_;
//     int instance_index_{-1};
// };

}  // namespace torchpipe