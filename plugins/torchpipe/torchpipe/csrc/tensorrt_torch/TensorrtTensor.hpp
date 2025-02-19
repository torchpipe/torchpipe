#pragma once
#include <hami/extension.hpp>
#include "NvInfer.h"

namespace torchpipe {

// Aspect[ModelLoadder[Onnx2Tensorrt, LoadTensorrtEngine] ->
// TensorrtInferTensor]
class TensorrtTensor : public hami::Backend {
   public:
    void init(const std::unordered_map<std::string, std::string>& config,
              const hami::dict& dict_config) override;
    void forward(const std::vector<hami::dict>& input) override;

   private:
    std::unique_ptr<nvinfer1::IRuntime> runtime_;
    std::shared_ptr<nvinfer1::ICudaEngine> engine_;
    int instance_index_{0};
    int instance_num_{1};
};

class Onnx2Tensorrt : public hami::Backend {
   public:
    void init(const std::unordered_map<std::string, std::string>& config,
              const hami::dict& dict_config) override;
    void forward(const std::vector<hami::dict>& input) override;

   private:
    std::unique_ptr<nvinfer1::IRuntime> runtime_;
    std::shared_ptr<nvinfer1::ICudaEngine> engine_;
    int instance_index_{0};
    int instance_num_{1};
};
}  // namespace torchpipe