#pragma once

#include <torch/torch.h>

namespace kvcache {
class KVCacheV4 {
 public:
  KVCacheV4(std::vector<std::vector<torch::Tensor>>&& kv) : kv_(kv) {}
  std::vector<torch::Tensor> next() {
    std::vector<torch::Tensor> tmp;
    std::swap(tmp, kv_[layer_index_++]);
    return tmp;
  }

 private:
  size_t layer_index_ = 0;
  std::vector<std::vector<torch::Tensor>> kv_;
};
}  // namespace kvcache