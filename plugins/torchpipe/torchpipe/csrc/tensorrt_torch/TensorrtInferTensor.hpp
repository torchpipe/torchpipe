#pragma once
#include <hami/extension.hpp>
#include "NvInfer.h"
// #include "ATen/cuda/CUDAEvent.h"
#include <cuda_runtime.h>
#include "helper/net_info.hpp"
#include <torch/torch.h>

namespace torchpipe {

class TensorrtInferTensor : public hami::Backend {
   public:
    void init(const std::unordered_map<std::string, std::string>& config,
              const hami::dict& dict_config) override;
    void forward(const std::vector<hami::dict>& input) override;
    [[nodiscard]] size_t max() const {
        return (size_t)info_.first.at(0).max.d[0];
    }
    [[nodiscard]] size_t min() const {
        return (size_t)info_.first.at(0).min.d[0];
    }

    ~TensorrtInferTensor();

   private:
    std::unique_ptr<nvinfer1::IRuntime> runtime_;
    std::shared_ptr<nvinfer1::ICudaEngine> engine_;
    int instance_index_{0};
    int instance_num_{1};

   private:
    std::unique_ptr<nvinfer1::IExecutionContext> context_;
    NetIOInfos info_;

   private:
    std::vector<torch::Tensor> inputs_;
    // std::vector<torch::Tensor> outputs_;
    // std::vector<void*> binding_;
    // std::vector<bool> should_change_shape_;
    size_t mem_size_ = 0;

    cudaEvent_t input_finish_event_{};
};

// class x : public hami::Backend {
//    public:
//     void init(const std::unordered_map<std::string, std::string>& config,
//               const hami::dict& dict_config) override;
//     void forward(const std::vector<hami::dict>& input) override;

//    private:
//     std::unique_ptr<nvinfer1::IRuntime> runtime_;
//     std::shared_ptr<nvinfer1::ICudaEngine> engine_;
//     int instance_index_{0};
//     int instance_num_{1};
// };
}  // namespace torchpipe