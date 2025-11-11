#pragma once

#include <omniback/extension.hpp>
#include <torch/torch.h>
#include <string>
#include <unordered_set>

using omniback::dict;

namespace torchpipe {

class IndexSelectTensor : public omniback::Backend {
 private:
  void impl_init(
      const std::unordered_map<std::string, std::string>& config_param,
      const dict& kwargs) override;
  void impl_forward(const std::vector<dict>& ios) override;

 protected:
  //   ArgsKwargs args_kwargs_;
  //   std::string device_ = "cuda";
  torch::Tensor weight_;
  torch::Device device_{"cuda"};

 private:
  virtual std::string reflect_cls_name() const {
    return "NotExistName";
  }
};

class EmbeddingTensor : public IndexSelectTensor {
  void impl_forward(const std::vector<dict>& ios) override;
  virtual std::string reflect_cls_name() const override {
    return "EmbeddingTensor";
  }
};
} // namespace torchpipe