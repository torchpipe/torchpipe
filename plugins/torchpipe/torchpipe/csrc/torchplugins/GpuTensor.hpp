#pragma once

#include <unordered_set>
#include <string>
#include <hami/extension.hpp>
#include <torch/torch.h>

using hami::dict;

namespace torchpipe {

class IndexSelectTensor : public hami::BackendOne {
 private:
  void impl_init(
      const std::unordered_map<std::string, std::string>& config_param,
      const dict& kwargs) override;
  void forward(const dict& input_output) override;

 protected:
  //   ArgsKwargs args_kwargs_;
  //   std::string device_ = "cuda";
  torch::Tensor weight_;
  torch::Device device_{"cuda"};

 private:
  virtual std::string default_cls_name() const {
    return "NotExistName";
  }
};

class EmbeddingTensor : public IndexSelectTensor {
  void forward(const dict& input_output) override;
  virtual std::string default_cls_name() const override {
    return "EmbeddingTensor";
  }
};
} // namespace torchpipe