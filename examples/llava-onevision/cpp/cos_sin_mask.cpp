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

#include <torch/torch.h>
#include <sstream>
#include "Backend.hpp"
#include "base_logging.hpp"
#include "params.hpp"
#include "exception.hpp"
#include "threadsafe_kv_storage.hpp"

namespace {
std::tuple<torch::Tensor, torch::Tensor> generate_simple_llama2_fp16_mask(int64_t sequence_length,
                                                                          int64_t target_length,
                                                                          bool is_prefill = true) {
  auto dtype = torch::kFloat16;
  auto device = torch::kCUDA;

  if (!is_prefill) {
    auto causal_mask = torch::zeros({1, 1, sequence_length, target_length},
                                    torch::TensorOptions().dtype(dtype).device(device));
    // SPDLOG_DEBUG("{} {}", sequence_length, target_length);
    auto cache_position = torch::arange(target_length - sequence_length, target_length,
                                        torch::TensorOptions().device(device))
                              .unsqueeze(0);
    return std::make_tuple(causal_mask, cache_position);
  }

  auto min_dtype = -65504.;  // torch.finfo(torch.float16).min

  auto cache_position = torch::arange(sequence_length, torch::TensorOptions().device(device));
  auto causal_mask = torch::full({sequence_length, target_length}, min_dtype,
                                 torch::TensorOptions().dtype(dtype).device(device));

  if (sequence_length != 1) {
    causal_mask = torch::triu(causal_mask, 1);
  }

  causal_mask *= (torch::arange(target_length, torch::TensorOptions().device(device)) >
                  cache_position.reshape({-1, 1}));
  causal_mask = causal_mask.unsqueeze(0).unsqueeze(0);
  cache_position = cache_position.unsqueeze(0);

  // base=10000.0
  // dim = 128
  //               inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2,
  //               dtype=torch.int64).float().to(device) / dim))
  // return inv_freq, attention_factor

  //   std::cout << causal_mask << std::endl;

  return std::make_tuple(causal_mask, cache_position);
}

class GeneralRotaryEmbedding {
  torch::Tensor inv_freq_;
  float attention_scaling_{1.0};

 public:
  // base=10000.0
  // dim = 128
  //               inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2,
  //               dtype=torch.int64).float().to(device) / dim))
  // return inv_freq, attention_factor
  GeneralRotaryEmbedding(double base, int64_t dim, torch::Device device) {
    // 初始化 inv_freq
    auto arange_tensor =
        torch::arange(0, dim, 2, torch::TensorOptions().dtype(torch::kInt64).device(device))
            .to(torch::kFloat32);
    inv_freq_ = 1.0 / torch::pow(base, arange_tensor / dim);
    inv_freq_ = inv_freq_.unsqueeze(0).unsqueeze(2).to(torch::kFloat32);

    // 初始化 attention_factor (假设它是一个标量)
  }
  std::tuple<torch::Tensor, torch::Tensor> forward(const torch::Tensor& x,
                                                   const torch::Tensor& position_ids) {
    // Core RoPE block
    auto inv_freq_expanded = inv_freq_.expand({position_ids.size(0), -1, 1});
    auto position_ids_expanded = position_ids.unsqueeze(1).to(torch::kFloat32);

    // Force float32 (see https://github.com/huggingface/transformers/pull/29285)
    // std::string device_type = x.device().str();
    // device_type = (device_type != "mps") ? device_type : "cpu";

    torch::Tensor freqs;
    {
      //   torch::autocast::Mode autocast_mode(device_type, false);
      freqs = torch::matmul(inv_freq_expanded, position_ids_expanded).transpose(1, 2);
    }

    auto emb = torch::cat({freqs, freqs}, -1);
    auto cos = emb.cos();
    auto sin = emb.sin();

    // Advanced RoPE types (e.g., yarn) apply a post-processing scaling factor, equivalent to
    // scaling attention
    if (attention_scaling_ != 1) {
      cos = cos * attention_scaling_;
      sin = sin * attention_scaling_;
    }

    return std::make_tuple(cos.to(x.dtype()), sin.to(x.dtype()));
  }
};
}  // namespace
namespace ipipe {

class AppendPrefillCosSinMaskTensor : public SingleBackend {
 public:
  bool init(const std::unordered_map<std::string, std::string>& config, dict dict_config) override {
    TRACE_EXCEPTION(dim_ = std::stoi(config.at("dim")));
    TRACE_EXCEPTION(base_ = std::stod(config.at("base")));
    IPIPE_ASSERT(dim_ >= 32);
    llamaRotaryEmbedding_ = std::make_unique<GeneralRotaryEmbedding>(base_, dim_, torch::kCUDA);
    return true;
  }
  void forward(dict input_dict) override {
    std::vector<torch::Tensor> data =
        dict_get<std::vector<torch::Tensor>>(input_dict, TASK_DATA_KEY);
    IPIPE_ASSERT(data.size() == 3, "input.size() must be 3 (qkv)");
    IPIPE_ASSERT(data[0].sizes().size() == 3, "input[0].sizes().size() must be 3, got " +
                                                  std::to_string(data[0].sizes().size()));
    auto seq_len = data[0].size(-2);
    if (cos_sin_mask_.size() != 3 || seq_len != cos_sin_mask_[0].size(-2)) {
      cos_sin_mask_.clear();

      auto target_length = seq_len;
      IPIPE_ASSERT(torch::kFloat16 == data[0].dtype(), "data.dtype() must be torch.float16");
      auto result = generate_simple_llama2_fp16_mask(seq_len, target_length);

      auto cs = llamaRotaryEmbedding_->forward(data[0], std::get<1>(result));
      cos_sin_mask_.push_back(std::get<0>(cs));
      cos_sin_mask_.push_back(std::get<1>(cs));
      cos_sin_mask_.push_back(std::get<0>(result));
    }
    data.insert(data.end(), cos_sin_mask_.begin(), cos_sin_mask_.end());
    // data.push_back(std::get<0>(cs));
    // data.push_back(std::get<1>(cs));
    // data.push_back(std::get<0>(result));
    input_dict->operator[](TASK_RESULT_KEY) = data;
  }

 private:
  std::unique_ptr<GeneralRotaryEmbedding> llamaRotaryEmbedding_;
  std::vector<torch::Tensor> cos_sin_mask_;
  int dim_{128};
  double base_{10000};
  //   llamaRotaryEmbedding.forward(causal_mask, cache_position);
};

IPIPE_REGISTER(Backend, AppendPrefillCosSinMaskTensor);

class AppendDecodeCosSinMaskTensor : public SingleBackend {
 public:
  bool init(const std::unordered_map<std::string, std::string>& config, dict dict_config) override {
    TRACE_EXCEPTION(dim_ = std::stoi(config.at("dim")));
    TRACE_EXCEPTION(base_ = std::stod(config.at("base")));
    IPIPE_ASSERT(dim_ >= 32);
    llamaRotaryEmbedding_ = std::make_unique<GeneralRotaryEmbedding>(base_, dim_, torch::kCUDA);
    return true;
  }
  void forward(dict input_dict) override {
    std::vector<torch::Tensor> data =
        dict_get<std::vector<torch::Tensor>>(input_dict, TASK_DATA_KEY);
    IPIPE_ASSERT(data.size() == 3, "input.size() must be 3 (qkv)");
    IPIPE_ASSERT(data[0].sizes().size() == 3, "input[0].sizes().size() must be 3, got " +
                                                  std::to_string(data[0].sizes().size()));
    auto seq_len = data[0].size(-2);

    auto iter = input_dict->find("request_id");
    IPIPE_ASSERT(iter != input_dict->end(), "request_id is needed");
    auto request_id = any_cast<std::string>(iter->second);
    static auto& storage = ThreadSafeKVStorage::getInstance().get(request_id);
    static auto token_counter = storage.get<std::shared_ptr<TypedDict>>("token_counter");

    auto target_length = std::get<int>(token_counter->data.at("new_tokens")) +
                         std::get<int>(token_counter->data.at("input_tokens"));

    if (cos_sin_mask_.size() != 3 || target_length != cos_sin_mask_[0].size(-2)) {
      cos_sin_mask_.clear();

      { SPDLOG_DEBUG("target_length: {}", target_length); }

      IPIPE_ASSERT(torch::kFloat16 == data[0].dtype(), "data.dtype() must be torch.float16");
      auto result = generate_simple_llama2_fp16_mask(seq_len, target_length, false);

      auto cs = llamaRotaryEmbedding_->forward(data[0], std::get<1>(result));
      cos_sin_mask_.push_back(std::get<0>(cs));
      cos_sin_mask_.push_back(std::get<1>(cs));
      cos_sin_mask_.push_back(std::get<0>(result));
    }
    data.insert(data.end(), cos_sin_mask_.begin(), cos_sin_mask_.end());
    // data.push_back(std::get<0>(cs));
    // data.push_back(std::get<1>(cs));
    // data.push_back(std::get<0>(result));
    input_dict->operator[](TASK_RESULT_KEY) = data;
  }

 private:
  std::unique_ptr<GeneralRotaryEmbedding> llamaRotaryEmbedding_;
  std::vector<torch::Tensor> cos_sin_mask_;
  int dim_{128};
  double base_{10000};
  //   llamaRotaryEmbedding.forward(causal_mask, cache_position);
};

IPIPE_REGISTER(Backend, AppendDecodeCosSinMaskTensor);

}  // namespace ipipe