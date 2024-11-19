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

#include <tuple>

#include "KVCacheTensorV3.hpp"
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
#include "threadsafe_kv_storage.hpp"
#include "exception.hpp"
#include "sampling_params.h"
#include "base_logging.hpp"
#include "KVCacheManagerBase.hpp"
#include "KVCacheTensorType.hpp"

namespace ipipe {

bool KVCacheIOTensor::init(const std::unordered_map<std::string, std::string>& config_param,
                           dict dict_config) {
  params_ =
      std::unique_ptr<Params>(new Params({{"KVCacheIOTensor::backend", "Identity"}}, {}, {}, {}));

  if (!params_->init(config_param)) return false;

  engine_ =
      std::unique_ptr<Backend>(IPIPE_CREATE(Backend, params_->at("KVCacheIOTensor::backend")));

  auto config_param_new = config_param;
  config_param_new.erase("KVCacheIOTensor::backend");

  if (!engine_ || !engine_->init(config_param_new, dict_config)) {
    return false;
  }

  return true;
}

void KVCacheIOTensor::forward(dict input_dict) {
  // if (kvcache_manager_ == nullptr) {
  //   auto& storage_local = ThreadSafeKVStorage::getInstance(ThreadSafeKVStorage::POOL::SCHEDULER);
  //   kvcache_manager_ =
  //       any_cast<kvcache::KVCacheManagerBase*>(storage_local.get("").get("kvcache_manager"));
  // }
  static auto& storage = ThreadSafeKVStorage::getInstance();

  auto iter = input_dict->find("request_id");
  IPIPE_ASSERT(iter != input_dict->end());
  std::string request_id = any_cast<std::string>(iter->second);
  // SPDLOG_DEBUG("KVCacheIOTensor request_id: {}", request_id);

  auto& storage_kv = storage.get(request_id);

  std::shared_ptr<kvcache::KVCacheV4> kvcache =
      any_cast<std::shared_ptr<kvcache::KVCacheV4>>(storage_kv.get("kvcachev4"));

  std::vector<torch::Tensor> kvcache_tensor = kvcache->next();
  // SPDLOG_INFO("KVCacheIOTensor kvcache_tensor.size() = {} ", kvcache_tensor.size());

  auto iter_kvcache = kvcache_tensor.begin();
  if (kvcache_tensor.size() == 4) {
    auto iter = input_dict->find(TASK_DATA_KEY);
    std::vector<torch::Tensor>* data = any_cast<std::vector<torch::Tensor>>(&iter->second);
    SPDLOG_DEBUG("KVCacheIOTensor(decode stage) original data.size() = {} ", data->size());
    data->insert(data->end(), kvcache_tensor.begin(), kvcache_tensor.begin() + 2);
    iter_kvcache = kvcache_tensor.begin() + 2;
  } else {
    IPIPE_ASSERT(kvcache_tensor.size() == 2);
  }

  iter = input_dict->find("outputs");
  std::vector<torch::Tensor>* outputs = any_cast<std::vector<torch::Tensor>>(&iter->second);
  auto& other = *outputs;
  SPDLOG_DEBUG("KVCacheIOTensor original outputs.size() = {} ", other.size());
  other.insert(other.end(), iter_kvcache, kvcache_tensor.end());

  engine_->forward({input_dict});
}

IPIPE_REGISTER(Backend, KVCacheIOTensor, "KVCacheIOTensor");

// RemoveKVCache

}  // namespace ipipe