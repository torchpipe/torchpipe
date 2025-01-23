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

// #include <torch/torch.h>
#include "any.hpp"
#include "Backend.hpp"
#include "dict.hpp"

#include "params.hpp"
#include "reflect.h"
#include "time_utils.hpp"
#include "torch_utils.hpp"
#include "threadsafe_queue.hpp"
#include "base_logging.hpp"
#include "threadsafe_kv_storage.hpp"
#include "exception.hpp"

#include "base_logging.hpp"

namespace ipipe {

class AppendIndexSelectTensor : public SingleBackend {
 public:
  /**
   * @param tensor_name 文件路径；
   *
   */
  virtual bool init(const std::unordered_map<std::string, std::string>& config_param,
                    dict) override {
    params_ = std::unique_ptr<Params>(new Params({{"value", "-1"}}, {}, {}, {}));
    if (!params_->init(config_param)) return false;
    value_ = std::stoi(params_->at("value"));
    if (value_ == 0) {
      tensor_cache_ = std::make_unique<torch::Tensor>(
          torch::tensor({value_}, torch::TensorOptions().dtype(torch::kLong).device("cuda")));
    }

    return true;
  }

  /**
   * @param TASK_RESULT_KEY 加载的tensor
   */
  virtual void forward(dict input_dict) override {
    std::vector<torch::Tensor> inputs = dict_gets<torch::Tensor>(input_dict, TASK_DATA_KEY);
    IPIPE_ASSERT(inputs.size() == 1);

    const auto& input = inputs[0];
    long index_select = input.sizes()[0];
    if (value_ < 0)
      index_select += value_;
    else {
      index_select = value_;
    }
    IPIPE_ASSERT(index_select < input.sizes()[0]);

    if (index_select == 0) {
      if (!tensor_cache_) {
        SPDLOG_INFO("index_select = {}", index_select);
        tensor_cache_ = std::make_unique<torch::Tensor>(
            torch::tensor({index_select}, torch::TensorOptions()
                                              .dtype(torch::kLong)
                                              .device(input[0].device())));  // todo: device check?
      }
      inputs.push_back(*tensor_cache_);
    } else {
      SPDLOG_DEBUG("index_select = {}", index_select);
      torch::Tensor index = torch::tensor(
          {index_select}, torch::TensorOptions().dtype(torch::kLong).device(input[0].device()));

      inputs.push_back(index);
    }

    (*input_dict)[TASK_RESULT_KEY] = inputs;
  }

 private:
  std::unique_ptr<Params> params_;
  // torch::Tensor tensor_;
  // torch::Device device_ = torch::Device{"cuda"};
  int value_{-1};
  std::unique_ptr<torch::Tensor> tensor_cache_;
};

IPIPE_REGISTER(Backend, AppendIndexSelectTensor, "AppendIndexSelectTensor");

class RemoveStorage : public SingleBackend {
 public:
  void forward(dict input) {
    auto iter = input->find("request_id");
    if (iter != input->end()) {
      auto request_id = any_cast<std::string>(iter->second);
      SPDLOG_DEBUG("RemoveStorage: {}", request_id);
      ThreadSafeKVStorage::getInstance().remove(request_id);
    }
    TRACE_EXCEPTION((*input)[TASK_RESULT_KEY] = (input)->at(TASK_DATA_KEY));
  }
};
IPIPE_REGISTER(Backend, RemoveStorage, "RemoveStorage");

class InitTokenCounter : public SingleBackend {
 public:
  void forward(dict input) {
    auto iter = input->find("request_id");
    IPIPE_ASSERT(iter != input->end(), "request_id is needed for InitTokenCounter");
    auto request_id = any_cast<std::string>(iter->second);

    auto& storage = ThreadSafeKVStorage::getInstance().get_or_insert(request_id);
    if (!storage.has("token_counter")) {
      int req_size = get_request_size(input);

      auto token_counter = std::shared_ptr<TypedDict>(new TypedDict());
      token_counter->data["input_tokens"] = req_size;
      token_counter->data["new_tokens"] = 0;

      storage.set("token_counter", token_counter);

      // auto sampling_params = dict_get<std::shared_ptr<TypedDict>>(input, "sampling_params");

      // TRACE_EXCEPTION(storage.set("max_seq_len", sampling_params->at("max_seq_len")));
      // TRACE_EXCEPTION(storage.set("max_tokens", sampling_params->at("max_tokens")));
      // TRACE_EXCEPTION(storage.set("max_tokens", sampling_params->at("stop_token_ids")));
    }
    TRACE_EXCEPTION((*input)[TASK_RESULT_KEY] = (input)->at(TASK_DATA_KEY));
  }
};
IPIPE_REGISTER(Backend, InitTokenCounter, "InitTokenCounter");

class UpdateTokenCounter : public SingleBackend {
 public:
  void forward(dict input) {
    auto iter = input->find("request_id");
    IPIPE_ASSERT(iter != input->end(), "request_id is needed for UpdateTokenCounter");

    auto request_id = any_cast<std::string>(iter->second);
    auto& storage = ThreadSafeKVStorage::getInstance().get(request_id);
    auto token_counter = storage.get<std::shared_ptr<TypedDict>>("token_counter");

    int& new_tokens = std::get<int>(token_counter->data.at("new_tokens"));
    new_tokens += 1;
    SPDLOG_DEBUG("UpdateTokenCounter: id={}, new_tokens={}", request_id, new_tokens);

    TRACE_EXCEPTION((*input)[TASK_RESULT_KEY] = (input)->at(TASK_DATA_KEY));
  }
};

IPIPE_REGISTER(Backend, UpdateTokenCounter, "UpdateTokenCounter");

class LLMRestart : public SingleBackend {
 public:
  virtual bool init(const std::unordered_map<std::string, std::string>& config_param,
                    dict) override {
    params_ = std::unique_ptr<Params>(new Params({}, {"restart"}, {}, {}));
    if (!params_->init(config_param)) return false;
    restart_node_ = (params_->at("restart"));

    return true;
  }
  void forward(dict input) {
    auto iter = input->find("request_id");
    IPIPE_ASSERT(iter != input->end(), "request_id is needed for LLMRestart");

    auto request_id = any_cast<std::string>(iter->second);

    std::shared_ptr<TypedDict> step_result = std::make_shared<TypedDict>();
    step_result->data["request_id"] = request_id;

    static auto& storage = ThreadSafeKVStorage::getInstance().get(request_id);
    static auto token_counter = storage.get<std::shared_ptr<TypedDict>>("token_counter");

    int new_tokens = std::get<int>(token_counter->data.at("new_tokens"));
    int all_tokens = std::get<int>(token_counter->data.at("input_tokens")) + new_tokens;

    auto sampling_params = dict_get<std::shared_ptr<TypedDict>>(input, "sampling_params");

    int max_seq_len = std::get<int>(sampling_params->data.at("max_seq_len"));
    int max_tokens = std::get<int>(sampling_params->data.at("max_tokens"));
    const std::vector<int>& stop_token_ids =
        std::get<std::vector<int>>(sampling_params->data.at("stop_token_ids"));
    torch::Tensor data = dict_get<torch::Tensor>(input, "data");
    int generated_token = data.item<int>();

    step_result->data["generated_token"] = generated_token;

    if (std::find(stop_token_ids.begin(), stop_token_ids.end(), generated_token) !=
        stop_token_ids.end()) {
      step_result->data["finish_reason"] = std::string("stop");
    } else if (all_tokens >= max_seq_len) {
      step_result->data["finish_reason"] = std::string("length");
    } else if (new_tokens >= max_tokens) {
      step_result->data["finish_reason"] = std::string("length");
    } else {
      (*input)["restart"] = restart_node_;
      step_result->data["finish_reason"] = std::string();
    }

    // auto& queue =
    //     storage.get_or_insert<ThreadSafeQueue<std::shared_ptr<TypedDict>>>("iteration");

    static auto& storage_global =
        ThreadSafeKVStorage::getInstance(ThreadSafeKVStorage::POOL::SCHEDULER)
            .get_or_insert("iteration");
    static auto queue =
        storage_global.get_or_insert<ThreadSafeQueue<std::shared_ptr<TypedDict>>>("queue");
    queue->Push(step_result);

    // auto& storage = ThreadSafeKVStorage::getInstance().get(request_id);
    // int* storage.get<int>("past_tokens");

    SPDLOG_DEBUG("LLMRestart/id={}, generated_token={}", request_id, generated_token);
    (*input)[TASK_RESULT_KEY] = (input)->at(TASK_DATA_KEY);

    // TRACE_EXCEPTION((*input)[TASK_RESULT_KEY] = (input)->at(TASK_DATA_KEY));
  }

 private:
  std::unique_ptr<Params> params_;
  std::string restart_node_;
};

IPIPE_REGISTER(Backend, LLMRestart, "LLMRestart");
}  // namespace ipipe