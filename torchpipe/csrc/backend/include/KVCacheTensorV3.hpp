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

namespace kvcache {
class KVCacheManagerBase;
}
namespace ipipe {
class Params;

// class KVCacheV3 {
//  public:
//   std::vector<torch::Tensor> pop() {
//     std::vector<torch::Tensor> tmp;
//     std::swap(tmp, kv_cache_.front());
//     kv_cache_.pop();
//     return tmp;
//   }

//   template <typename T>
//   void push(T&& input) {
//     kv_cache_.push(std::forward<T>(input));
//     // ++current_layer_;
//   }

//   std::size_t size() { return kv_cache_.size(); }

//  private:
//   std::queue<std::vector<torch::Tensor>> kv_cache_;  // 32x2: layer index && k,v
//   std::size_t current_layer_ = 0;
// };

// class PushAndErase : public SingleBackend {
//  public:
//   virtual bool init(const std::unordered_map<std::string, std::string>&, dict) override;

//   virtual void forward(dict) override;

//  private:
//   std::unique_ptr<Params> params_;
//   std::unique_ptr<Backend> engine_;
//   std::vector<std::string> keys_;
// };

// class PopKVCacheTensorV3 : public SingleBackend {
//  public:
//   virtual bool init(const std::unordered_map<std::string, std::string>&, dict) override;

//   virtual void forward(dict) override;

//  private:
//   std::unique_ptr<Params> params_;
//   std::unique_ptr<Backend> engine_;
// };

class KVCacheIOTensor : public SingleBackend {
 public:
  virtual bool init(const std::unordered_map<std::string, std::string>&, dict) override;

  virtual void forward(dict) override;

 private:
  std::unique_ptr<Params> params_;
  std::unique_ptr<Backend> engine_;
};

}  // namespace ipipe
