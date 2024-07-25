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

#include "KVCacheTensor.hpp"
#include "Sequential.hpp"

// #include <torch/torch.h>
#include "any.hpp"
#include "Backend.hpp"
#include "dict.hpp"
#include "c10/cuda/CUDAStream.h"
#include "dict_helper.hpp"
#include "params.hpp"
#include "reflect.h"
#include "time_utils.hpp"
#include "torch_utils.hpp"
#include "cuda_runtime_api.h"
#include "base_logging.hpp"
namespace ipipe {
bool KVCacheTensor::init(const std::unordered_map<std::string, std::string>& config_param,
                         dict dict_config) {
  params_ = std::unique_ptr<Params>(
      new Params({{"KVCacheTensor::backend", "Identity"}, {"num_layers", "32"}}, {}, {}, {}));

  if (!params_->init(config_param)) return false;

  num_layers_ = std::stoi(params_->at("num_layers"));

  engine_ = std::unique_ptr<Backend>(IPIPE_CREATE(Backend, params_->at("KVCacheTensor::backend")));

  auto config_param_new = config_param;
  config_param_new.erase("KVCacheTensor::backend");

  if (!engine_ || !engine_->init(config_param_new, dict_config)) {
    return false;
  }

  auto options = torch::TensorOptions()
                     .device(torch::kCUDA, -1)
                     .dtype(torch::kLong)
                     .layout(torch::kStrided)
                     .requires_grad(false);
  std::vector<long int> shape_{1, 2048};
  auto seq_length = shape_[1];
  position_ids_ = torch::arange(0, seq_length, options);

  return true;
}

void KVCacheTensor::forward(dict input_dict) {
  auto iter = input_dict->find("request_id");
  IPIPE_ASSERT(iter != input_dict->end());
  std::string request_id = any_cast<std::string>(iter->second);
  SPDLOG_DEBUG("KVCacheTensor request_id: {}", request_id);

  auto iter_remove = input_dict->find("remove_request_id");
  if (iter_remove != input_dict->end()) {
    SPDLOG_INFO("KVCacheTensor remove_request_id: {}", request_id);
    std::lock_guard<std::mutex> lock(mutex_);
    auto iter_cache = (kv_caches_.find(request_id));
    IPIPE_ASSERT(iter_cache != kv_caches_.end());
    kv_caches_.erase(iter_cache);
    return;
  }

  auto& input = *input_dict;
  std::vector<torch::Tensor>* input_tensor = nullptr;
  if (input[TASK_DATA_KEY].type() == typeid(torch::Tensor)) {
    // todo
    throw std::runtime_error(
        "KVCacheTensor: std::vector<torch::Tensor> needed; error input type: " +
        std::string(input[TASK_DATA_KEY].type().name()));
  } else if (input[TASK_DATA_KEY].type() == typeid(std::vector<torch::Tensor>)) {
    input_tensor = any_cast<std::vector<torch::Tensor>>(&input[TASK_DATA_KEY]);

  } else {
    throw std::runtime_error("vector<torch::Tensor> needed; error input type: " +
                             std::string(input[TASK_DATA_KEY].type().name()));
  }

  IPIPE_ASSERT(input_tensor->size() == 3);
  int seq_len = input_tensor->at(0).size(-2);
  int kv_seq_len = input_tensor->at(1).size(-2);
  IPIPE_ASSERT(seq_len == kv_seq_len || (seq_len == 1));
  SPDLOG_DEBUG("KVCache: seq_len: {} kv_seq_len {}", seq_len, kv_seq_len);

  KVCache* cache = nullptr;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    auto iter_kvcache = kv_caches_.find(request_id);
    if (iter_kvcache == kv_caches_.end()) {
      kv_caches_[request_id] = std::make_unique<KVCache>(num_layers_);
    }
    cache = kv_caches_[request_id].get();
  }
  auto state = cache->get_and_switch_state();
  if (cache->is_prefill()) {
    if (state == KVCache::KVCacheState::kPrepareInput) {
      torch::Tensor position_ids =
          position_ids_.index({torch::indexing::Slice(0, seq_len)}).unsqueeze(0);
      input_tensor->push_back(position_ids);
    } else {
      std::vector<torch::Tensor> kv(input_tensor->end() - 2, input_tensor->end());
      cache->push(kv);
    }
  } else {
    if (state == KVCache::KVCacheState::kPrepareInput) {
      auto past_kv = cache->pop();
      int past_seq_len = past_kv.at(0).size(-2);
      torch::Tensor position_ids =
          position_ids_.index({torch::indexing::Slice(past_seq_len, 1 + past_seq_len)})
              .unsqueeze(0);
      input_tensor->push_back(position_ids);
      input_tensor->push_back(past_kv.at(0));
      input_tensor->push_back(past_kv.at(1));
    } else {
      std::vector<torch::Tensor> kv(input_tensor->end() - 2, input_tensor->end());
      cache->push(kv);
    }
  }

  engine_->forward({input_dict});
}

IPIPE_REGISTER(Backend, KVCacheTensor, "KVCacheTensor");

}  // namespace ipipe