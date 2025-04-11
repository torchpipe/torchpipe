#pragma once

#include <unordered_set>
#include <string>
#include <hami/extension.hpp>
#include <torch/torch.h>

using hami::dict;

namespace torchpipe {

class IndexSelectTensor : public hami::Backend {
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
  virtual std::string default_cls_name() const {
    return "NotExistName";
  }
};

class EmbeddingTensor : public IndexSelectTensor {
  void impl_forward(const std::vector<dict>& ios) override;
  virtual std::string default_cls_name() const override {
    return "EmbeddingTensor";
  }
};
} // namespace torchpipe