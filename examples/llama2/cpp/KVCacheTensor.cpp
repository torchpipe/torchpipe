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

  // auto instance_num = std::stoi(params_->at("instance_num"));
  // IPIPE_ASSERT(instance_num == 1);

  // todo
  //  int _independent_thread_index = 0;

  //  if (!params_->at("_independent_thread_index").empty()) {
  //    TRACE_EXCEPTION(_independent_thread_index =
  //                        std::stoi(params_->at("_independent_thread_index")));
  //  } else {
  //    _independent_thread_index = 0;
  //  }

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

  auto kvcache = storage_kv.get("kvcache");
  std::shared_ptr<KVCacheV2> pkvcache;
  // std::lock_guard<std::mutex> lock(mutex_);
  if (!kvcache) {
    pkvcache = std::make_shared<KVCacheV2>();
    SPDLOG_DEBUG("kvcache set {} {}", (void*)pkvcache.get(), (void*)&storage_kv);
    storage_kv.set("kvcache", pkvcache);
  } else {
    pkvcache = any_cast<std::shared_ptr<ipipe::KVCacheV2>>(*kvcache);
    SPDLOG_DEBUG("kvcache get {} {}", (void*)pkvcache.get(), (void*)&storage_kv);
  }

  KVCacheV2* cache = pkvcache.get();

  IPIPE_ASSERT(input_tensor->size() >= 2);
  std::vector<torch::Tensor> kv(input_tensor->end() - 2, input_tensor->end());
  // final_past_seq_len = kv.at(0).size(-2);

  cache->push(std::move(kv));

  // (*input_dict)[TASK_RESULT_KEY] = (*input_dict)[TASK_DATA_KEY];

  // if (cache->round_over()) {
  //   if (final_past_seq_len >= max_seq_len_ - 1) {
  //     SPDLOG_DEBUG("PrefillPushKVCacheTensor: is eos");
  //     storage_kv.erase("kvcache");
  //     storage_kv.set("is_eos", 1);
  //   }
  // }

  engine_->forward({input_dict});
}

IPIPE_REGISTER(Backend, PushKVCacheTensor, "PushKVCacheTensor");

bool PopKVCacheTensor::init(const std::unordered_map<std::string, std::string>& config_param,
                            dict dict_config) {
  // params_ = std::unique_ptr<Params>(new Params({}, {}, {}, {}));

  // if (!params_->init(config_param)) return false;

  return true;
}

void PopKVCacheTensor::forward(dict input_dict) {
  static auto& storage = ThreadSafeKVStorage::getInstance();

  auto iter = input_dict->find("request_id");
  IPIPE_ASSERT(iter != input_dict->end());
  std::string request_id = any_cast<std::string>(iter->second);
  SPDLOG_DEBUG("PopgKVCacheTensor request_id: {}", request_id);

  auto& storage_kv = storage.get_or_insert(request_id);

  auto& input = *input_dict;

  auto kvcache = storage_kv.get("kvcache");
  IPIPE_ASSERT(kvcache);
  std::shared_ptr<KVCacheV2> pkvcache = any_cast<std::shared_ptr<ipipe::KVCacheV2>>(*kvcache);
  IPIPE_ASSERT(pkvcache);

  std::vector<torch::Tensor> kv = pkvcache->pop();
  IPIPE_ASSERT(kv.size() == 2);
  // final_past_seq_len = kv.at(0).size(-2);

  (*input_dict)[TASK_RESULT_KEY] = kv;
}

IPIPE_REGISTER(Backend, PopKVCacheTensor, "PopKVCacheTensor");
// RemoveKVCache

class RemoveKVCache : public SingleBackend {
 public:
  void forward(dict input) {
    auto iter = input->find("request_id");
    IPIPE_ASSERT(iter != input->end());
    {
      auto request_id = any_cast<std::string>(iter->second);
      SPDLOG_INFO("RemoveKVCache: {}", request_id);
      ThreadSafeKVStorage::getInstance().get(request_id).erase("kvcache");
    }
    (*input)[TASK_RESULT_KEY] = (*input)[TASK_DATA_KEY];
  }
};
IPIPE_REGISTER(Backend, RemoveKVCache, "RemoveKVCache");

class RemoveStorage : public SingleBackend {
 public:
  void forward(dict input) {
    auto iter = input->find("request_id");
    IPIPE_ASSERT(iter != input->end());
    {
      auto request_id = any_cast<std::string>(iter->second);
      SPDLOG_INFO("RemoveStorage: {}", request_id);
      ThreadSafeKVStorage::getInstance().erase(request_id);
    }
    TRACE_EXCEPTION((*input)[TASK_RESULT_KEY] = (input)->at(TASK_DATA_KEY));
  }
};
IPIPE_REGISTER(Backend, RemoveStorage, "RemoveStorage");

class CheckOtherSeqLenTensor : public SingleBackend {
 private:
  std::unique_ptr<Params> params_;

  std::string other_;
  int max_seq_len_;
  int max_new_tokens_;

 public:
  bool init(const std::unordered_map<std::string, std::string>& config_param,
            dict dict_config) override {
    params_ = std::unique_ptr<Params>(new Params(
        {{"other", "other"}, {"max_seq_len", "-1"}, {"max_new_tokens", "-1"}}, {}, {}, {}));
    if (!params_->init(config_param)) return false;
    other_ = params_->at("other");
    max_seq_len_ = std::stoi(params_->at("max_seq_len"));
    max_new_tokens_ = std::stoi(params_->at("max_new_tokens"));
    if (max_seq_len_ <= 0) {
      max_seq_len_ = INT32_MAX;
    }
    if (max_new_tokens_ <= 0) {
      max_new_tokens_ = INT32_MAX;
    }
    return true;
  }

  void forward(dict input_dict) override {
    auto other = dict_gets<torch::Tensor>(input_dict, other_);
    IPIPE_ASSERT(other.size() >= 2);
    int new_tokens = other.size() - 1;
    int seq_len = other[0].size(0) + new_tokens;
    SPDLOG_DEBUG("CheckOtherSeqLenTensor: seq_len: {}, new_tokens: {}", seq_len, new_tokens);
    if (seq_len >= max_seq_len_ || new_tokens >= max_new_tokens_) {
      input_dict->erase(other_);
    }
    (*input_dict)[TASK_RESULT_KEY] = input_dict->operator[](TASK_DATA_KEY);
  }
};
IPIPE_REGISTER(Backend, CheckOtherSeqLenTensor, "CheckOtherSeqLenTensor");

}  // namespace ipipe