// Copyright 2021-2024 NetEase.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include "Backend.hpp"
#include "dict.hpp"

#include <memory>
#include <torch/torch.h>

namespace ipipe {
class Params;
// class KVCache {
//  public:
//   enum class KVCacheState { kPrepareInput, kUpdateoutput };

//   KVCache(int num_layers) : num_layer_(num_layers) { kv_cache_.resize(num_layers); }

//   KVCacheState get_and_switch_state() {
//     if (state_ == KVCacheState::kPrepareInput) {
//       state_ = KVCacheState::kUpdateoutput;
//       return KVCacheState::kPrepareInput;
//     }
//     state_ = KVCacheState::kPrepareInput;
//     return KVCacheState::kUpdateoutput;
//   }
//   bool is_prefill() { return current_layer_ < num_layer_; }

//   std::vector<torch::Tensor> pop() {
//     std::vector<torch::Tensor> tmp;
//     std::swap(tmp, kv_cache_[current_layer_ % num_layer_]);
//     return tmp;
//   }
//   // const std::vector<torch::Tensor>& current() { return kv_cache_[current_layer_ % num_layer_];
//   } void push(std::vector<torch::Tensor> input) {
//     std::swap(kv_cache_[(current_layer_++) % num_layer_], input);
//   }

//   void push(std::vector<torch::Tensor> input) {
//     std::swap(kv_cache_[(current_layer_++) % num_layer_], input);
//   }

//   bool round_over() { return current_layer_ % num_layer_ == 0; }

//   std::size_t get_current_layer() { return current_layer_; }

//  private:
//   KVCacheState state_ = KVCacheState::kPrepareInput;
//   std::vector<std::vector<torch::Tensor>> kv_cache_;  // 32x2: layer index && k,v
//   std::size_t current_layer_ = 0;
//   const int num_layer_;
// };

class KVCacheV2 {
 public:
  std::vector<torch::Tensor> pop() {
    std::vector<torch::Tensor> tmp;
    std::swap(tmp, kv_cache_.front());
    kv_cache_.pop();
    return tmp;
  }

  template <typename T>
  void push(T&& input) {
    kv_cache_.push(std::forward<T>(input));
    // ++current_layer_;
  }

  std::size_t size() { return kv_cache_.size(); }

 private:
  std::queue<std::vector<torch::Tensor>> kv_cache_;  // 32x2: layer index && k,v
  std::size_t current_layer_ = 0;
};

class KVCacheTensor : public SingleBackend {
 public:
  virtual bool init(const std::unordered_map<std::string, std::string>&, dict) override;

  virtual void forward(dict) override;

 private:
  std::unique_ptr<Params> params_;
  std::unique_ptr<Backend> engine_;
  // layer_index -> request_id -> k,v
  // std::mutex mutex_;  // todo : support multiple instances
  torch::Tensor position_ids_;

  int num_layers_{0};
  int max_seq_len_{0};
};

class PushKVCacheTensor : public SingleBackend {
 public:
  virtual bool init(const std::unordered_map<std::string, std::string>&, dict) override;

  virtual void forward(dict) override;

 private:
  std::unique_ptr<Params> params_;
  std::unique_ptr<Backend> engine_;
};

class PushAndErase : public SingleBackend {
 public:
  virtual bool init(const std::unordered_map<std::string, std::string>&, dict) override;

  virtual void forward(dict) override;

 private:
  std::unique_ptr<Params> params_;
  std::unique_ptr<Backend> engine_;
  std::vector<std::string> keys_;
};

class PopKVCacheTensor : public SingleBackend {
 public:
  virtual bool init(const std::unordered_map<std::string, std::string>&, dict) override;

  virtual void forward(dict) override;

 private:
  std::unique_ptr<Params> params_;
  std::unique_ptr<Backend> engine_;
};

class RequestTimeStamp : public SingleBackend {
 public:
  virtual bool init(const std::unordered_map<std::string, std::string>&, dict) override;

  virtual void forward(dict) override;

 private:
  std::unique_ptr<Params> params_;
  std::unique_ptr<Backend> engine_;
  std::string key_;
};

}  // namespace ipipe
