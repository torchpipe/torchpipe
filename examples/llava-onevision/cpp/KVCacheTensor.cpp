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
#include "threadsafe_kv_storage.hpp"
#include "exception.hpp"
#include "sampling_params.h"
#include "base_logging.hpp"

namespace ipipe {
bool PushKVCacheTensor::init(const std::unordered_map<std::string, std::string>& config_param,
                             dict dict_config) {
  params_ =
      std::unique_ptr<Params>(new Params({{"PushKVCacheTensor::backend", "Identity"}}, {}, {}, {}));

  if (!params_->init(config_param)) return false;

  engine_ =
      std::unique_ptr<Backend>(IPIPE_CREATE(Backend, params_->at("PushKVCacheTensor::backend")));

  auto config_param_new = config_param;
  config_param_new.erase("PushKVCacheTensor::backend");

  if (!engine_ || !engine_->init(config_param_new, dict_config)) {
    return false;
  }

  return true;
}

void PushKVCacheTensor::forward(dict input_dict) {
  static auto& storage = ThreadSafeKVStorage::getInstance();

  auto iter = input_dict->find("request_id");
  IPIPE_ASSERT(iter != input_dict->end());
  std::string request_id = any_cast<std::string>(iter->second);
  SPDLOG_DEBUG("PushKVCacheTensor request_id: {}", request_id);

  auto& storage_kv = storage.get_or_insert(request_id);

  auto& input = *input_dict;
  std::vector<torch::Tensor>* input_tensor = nullptr;
  input_tensor = any_cast<std::vector<torch::Tensor>>(&input[TASK_DATA_KEY]);

  IPIPE_ASSERT(input_tensor->size() == 3);
  int seq_len = input_tensor->at(0).size(-2);
  int kv_seq_len = input_tensor->at(1).size(-2);
  IPIPE_ASSERT(seq_len == kv_seq_len || (seq_len == 1));
  SPDLOG_DEBUG("KVCache: seq_len: {} kv_seq_len {}", seq_len, kv_seq_len);

  int final_past_seq_len = -1;

  std::shared_ptr<KVCacheV2> pkvcache;
  // std::lock_guard<std::mutex> lock(mutex_);
  if (!storage_kv.has("kvcache")) {
    pkvcache = std::make_shared<KVCacheV2>();
    SPDLOG_DEBUG("kvcache set {} {}", (void*)pkvcache.get(), (void*)&storage_kv);
    storage_kv.set("kvcache", pkvcache);
  } else {
    auto kvcache = storage_kv.get("kvcache");
    pkvcache = any_cast<std::shared_ptr<ipipe::KVCacheV2>>(kvcache);
    // SPDLOG_DEBUG("kvcache get {} {}", (void*)pkvcache.get(), (void*)&storage_kv);
  }

  KVCacheV2* cache = pkvcache.get();

  IPIPE_ASSERT(input_tensor->size() >= 2);
  std::vector<torch::Tensor> kv(input_tensor->end() - 2, input_tensor->end());
  // final_past_seq_len = kv.at(0).size(-2);

  cache->push(std::move(kv));
  SPDLOG_DEBUG("kv cache size after 'push': {}", cache->size());

  engine_->forward({input_dict});
}

IPIPE_REGISTER(Backend, PushKVCacheTensor, "PushKVCacheTensor");

bool AppendKVCacheTensor::init(const std::unordered_map<std::string, std::string>& config_param,
                               dict dict_config) {
  // params_ = std::unique_ptr<Params>(new Params({}, {}, {}, {}));

  // if (!params_->init(config_param)) return false;

  return true;
}

void AppendKVCacheTensor::forward(dict input_dict) {
  static auto& storage = ThreadSafeKVStorage::getInstance();

  auto iter = input_dict->find("request_id");
  IPIPE_ASSERT(iter != input_dict->end());
  std::string request_id = any_cast<std::string>(iter->second);

  auto& storage_kv = storage.get(request_id);

  std::vector<torch::Tensor> input_tensors = dict_gets<torch::Tensor>(input_dict, TASK_DATA_KEY);

  auto& input = *input_dict;

  auto kvcache = storage_kv.get("kvcache");
  // IPIPE_ASSERT(kvcache);
  std::shared_ptr<KVCacheV2> pkvcache = any_cast<std::shared_ptr<ipipe::KVCacheV2>>(kvcache);
  IPIPE_ASSERT(pkvcache);

  std::vector<torch::Tensor> kv = pkvcache->pop();
  IPIPE_ASSERT(kv.size() == 2);
  input_tensors.insert(input_tensors.end(), kv.begin(), kv.end());
  SPDLOG_DEBUG("AppendKVCacheTensor request_id: {} seq_len: {}", request_id, kv[0].size(-2));

  (*input_dict)[TASK_RESULT_KEY] = input_tensors;
}

IPIPE_REGISTER(Backend, AppendKVCacheTensor, "AppendKVCacheTensor");
// RemoveKVCache

class RemoveStorage : public SingleBackend {
 public:
  void forward(dict input) {
    auto iter = input->find("request_id");
    IPIPE_ASSERT(iter != input->end());
    {
      auto request_id = any_cast<std::string>(iter->second);
      SPDLOG_INFO("RemoveStorage: {}", request_id);
      ThreadSafeKVStorage::getInstance().remove(request_id);
    }
    TRACE_EXCEPTION((*input)[TASK_RESULT_KEY] = (input)->at(TASK_DATA_KEY));
  }
};
IPIPE_REGISTER(Backend, RemoveStorage, "RemoveStorage");

bool RequestTimeStamp::init(const std::unordered_map<std::string, std::string>& config_param,
                            dict dict_config) {
  params_ = std::unique_ptr<Params>(
      new Params({{"RequestTimeStamp::backend", "Identity"}, {"key", ""}}, {}, {}, {}));

  if (!params_->init(config_param)) return false;

  engine_ =
      std::unique_ptr<Backend>(IPIPE_CREATE(Backend, params_->at("RequestTimeStamp::backend")));

  key_ = params_->at("key");
  auto config_param_new = config_param;
  config_param_new.erase("RequestTimeStamp::backend");

  if (!engine_ || !engine_->init(config_param_new, dict_config)) {
    return false;
  }

  return true;
}

void RequestTimeStamp::forward(dict input_dict) {
  static auto& storage = ThreadSafeKVStorage::getInstance();

  auto iter = input_dict->find("request_id");
  IPIPE_ASSERT(iter != input_dict->end());
  std::string request_id = any_cast<std::string>(iter->second);
  // SPDLOG_DEBUG("RequestTimeStamp request_id: {}", request_id);

  auto& storage_kv = storage.get_or_insert(request_id);

  auto& input = *input_dict;

  auto now_time = time_passed();
  if (!storage_kv.has("time_stamp")) {
    storage_kv.set("time_stamp", std::make_shared<std::vector<decltype(now_time)>>(1, now_time));
  } else {
    auto time_stamp = storage_kv.get("time_stamp");
    std::shared_ptr<std::vector<decltype(now_time)>> ptime_stamp =
        any_cast<std::shared_ptr<std::vector<decltype(now_time)>>>(time_stamp);
    if (key_.empty()) {
      SPDLOG_INFO("request({}, {}) time: {}", ptime_stamp->size() - 1, request_id,
                  now_time - ptime_stamp->back());
    } else {
      SPDLOG_INFO("request(key={}, index={}, {}) time: {}", key_, ptime_stamp->size() - 1,
                  request_id, now_time - ptime_stamp->back());
    }
    ptime_stamp->push_back(now_time);
  }

  engine_->forward({input_dict});
}

IPIPE_REGISTER(Backend, RequestTimeStamp, "RequestTimeStamp");

}  // namespace ipipe