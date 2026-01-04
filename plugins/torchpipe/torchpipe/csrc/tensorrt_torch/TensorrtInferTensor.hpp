#pragma once
#include <omniback/extension.hpp>
#include "NvInfer.h"
// #include "ATen/cuda/CUDAEvent.h"
#include <cuda_runtime.h>
#include <torch/torch.h>
#include "helper/net_info.hpp"
#include <omniback/addons/torch/type_traits.h>

namespace torchpipe {

class TensorrtInferTensor : public omniback::Backend {
 private:
  void impl_init(
      const std::unordered_map<std::string, std::string>& config,
      const omniback::dict& kwargs) override;
  void impl_forward(const std::vector<omniback::dict>& input) override;
  [[nodiscard]] uint32_t impl_max() const {
    return (size_t)info_.first.at(0).max.d[0];
  }
  [[nodiscard]] uint32_t impl_min() const {
    return (size_t)info_.first.at(0).min.d[0];
  }

  ~TensorrtInferTensor();

 private:
  std::unique_ptr<nvinfer1::IRuntime> runtime_;
  nvinfer1::ICudaEngine* engine_{nullptr};
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

} // namespace torchpipe