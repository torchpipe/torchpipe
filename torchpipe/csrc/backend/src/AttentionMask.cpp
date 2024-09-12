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
namespace {
std::tuple<torch::Tensor, torch::Tensor> generate_prefill_fp16_mask(int64_t sequence_length,
                                                                    int64_t target_length) {
  // torch.finfo(torch.float16).min
  auto min_dtype = -65504.;
  auto dtype = torch::kFloat16;
  auto device = torch::kCUDA;

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

class LlamaRotaryEmbedding {
  torch::Tensor inv_freq_;
  float attention_scaling_{1.0};

 public:
  // base=10000.0
  // dim = 128
  //               inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2,
  //               dtype=torch.int64).float().to(device) / dim))
  // return inv_freq, attention_factor
  LlamaRotaryEmbedding(double base, int64_t dim, torch::Device device) {
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

// class PrefillAttentionMask : public SingleBackend {
//  public:
//   void forward(dict input_dict) override {
//     torch::Tensor data = dict_get<torch::Tensor>(input_dict, TASK_DATA_KEY);
//     auto sequence_length = data.size(-2);
//     auto target_length = sequence_length;
//     IPIPE_ASSERT(torch::kFloat16 == data.dtype(), "data.dtype() must be torch.float16");
//     auto result = generate_prefill_fp16_mask(sequence_length, target_length);

//     auto cs = llamaRotaryEmbedding_->forward(data, std::get<1>(result));
//     input_dict->operator[](TASK_RESULT_KEY) =
//         std::make_tuple(std::get<0>(result), std::get<0>(cs), std::get<1>(cs));
//     std::cout << "PrefillAttentionMask forward" << std::get<0>(result) << std::get<1>(result)
//               << std::endl;
//   }
//   bool init(const std::unordered_map<std::string, std::string>& config, dict dict_config)
//   override {
//     llamaRotaryEmbedding_ = std::make_unique<LlamaRotaryEmbedding>(10000.0, 128, torch::kCUDA);
//     return true;
//   }

//  private:
//   std::unique_ptr<LlamaRotaryEmbedding> llamaRotaryEmbedding_;
//   //   llamaRotaryEmbedding.forward(causal_mask, cache_position);
// };

// IPIPE_REGISTER(Backend, PrefillAttentionMask);

class AppendCosSinMaskTensor : public SingleBackend {
 public:
  void forward(dict input_dict) override {
    std::vector<torch::Tensor> data =
        dict_get<std::vector<torch::Tensor>>(input_dict, TASK_DATA_KEY);
    IPIPE_ASSERT(data.size() == 3, "input.size() must be 3 (qkv)");
    IPIPE_ASSERT(data[0].sizes().size() == 3, "input[0].sizes().size() must be 3, got " +
                                                  std::to_string(data[0].sizes().size()));
    auto seq_len = data[0].size(-2);
    auto target_length = seq_len;
    IPIPE_ASSERT(torch::kFloat16 == data[0].dtype(), "data.dtype() must be torch.float16");
    auto result = generate_prefill_fp16_mask(seq_len, target_length);

    auto cs = llamaRotaryEmbedding_->forward(data[0], std::get<1>(result));
    data.push_back(std::get<0>(cs));
    data.push_back(std::get<1>(cs));
    data.push_back(std::get<0>(result));
    input_dict->operator[](TASK_RESULT_KEY) = data;
  }
  bool init(const std::unordered_map<std::string, std::string>& config, dict dict_config) override {
    llamaRotaryEmbedding_ = std::make_unique<LlamaRotaryEmbedding>(10000.0, 128, torch::kCUDA);
    return true;
  }

 private:
  std::unique_ptr<LlamaRotaryEmbedding> llamaRotaryEmbedding_;
  //   llamaRotaryEmbedding.forward(causal_mask, cache_position);
};

IPIPE_REGISTER(Backend, AppendCosSinMaskTensor);

class GenerateCosSinMaskTensor : public SingleBackend {
 public:
  void forward(dict input_dict) override {
    torch::Tensor data = dict_get<torch::Tensor>(input_dict, TASK_DATA_KEY);

    auto seq_len = data.size(-2);
    auto target_length = seq_len;
    auto result = generate_prefill_fp16_mask(seq_len, target_length);

    auto cs = llamaRotaryEmbedding_->forward(data[0], std::get<1>(result));
    std::vector<torch::Tensor> final_result;
    final_result.push_back(std::get<0>(cs));
    final_result.push_back(std::get<1>(cs));
    final_result.push_back(std::get<0>(result));
    input_dict->operator[](TASK_RESULT_KEY) = final_result;
  }
  bool init(const std::unordered_map<std::string, std::string>& config, dict dict_config) override {
    llamaRotaryEmbedding_ = std::make_unique<LlamaRotaryEmbedding>(10000.0, 128, torch::kCUDA);
    return true;
  }

 private:
  std::unique_ptr<LlamaRotaryEmbedding> llamaRotaryEmbedding_;
  //   llamaRotaryEmbedding.forward(causal_mask, cache_position);
};

IPIPE_REGISTER(Backend, GenerateCosSinMaskTensor);

class AppendOtherTensor : public SingleBackend {
 private:
  std::string other_;
  std::unique_ptr<Params> params_;

 public:
  bool init(const std::unordered_map<std::string, std::string>& config_param,
            dict dict_config) override {
    params_ = std::unique_ptr<Params>(new Params({{"other", "other"}}, {}, {}, {}));
    if (!params_->init(config_param)) return false;
    other_ = params_->at("other");
    // backend_ = std::unique_ptr<Backend>(IPIPE_CREATE(Backend,
    // params_->at("Result2Key::backend")));
    // if (!backend_ || !backend_->init(config_param, dict_config)) return false;
    return true;
  }
  void forward(dict input_dict) override {
    auto data = dict_gets<torch::Tensor>(input_dict, TASK_DATA_KEY);

    auto other = dict_gets<torch::Tensor>(input_dict, other_);

    data.insert(data.end(), other.begin(), other.end());
    input_dict->operator[](TASK_RESULT_KEY) = data;
  }

 private:
  //   llamaRotaryEmbedding.forward(causal_mask, cache_position);
};

IPIPE_REGISTER(Backend, AppendOtherTensor);

class Cache2OtherTensor : public SingleBackend {
 private:
  std::string other_;
  std::unique_ptr<Params> params_;

 public:
  bool init(const std::unordered_map<std::string, std::string>& config_param,
            dict dict_config) override {
    params_ = std::unique_ptr<Params>(new Params({{"other", "other"}}, {}, {}, {}));
    if (!params_->init(config_param)) return false;
    other_ = params_->at("other");
    IPIPE_ASSERT(other_ != TASK_RESULT_KEY);
    // backend_ = std::unique_ptr<Backend>(IPIPE_CREATE(Backend,
    // params_->at("Result2Key::backend")));
    // if (!backend_ || !backend_->init(config_param, dict_config)) return false;
    return true;
  }
  void forward(dict input_dict) override {
    auto data = dict_gets<torch::Tensor>(input_dict, TASK_DATA_KEY);

    auto other = dict_gets<torch::Tensor>(input_dict, other_);

    other.insert(other.end(), data.begin(), data.end());
    input_dict->operator[](other_) = other;
    input_dict->operator[](TASK_RESULT_KEY) = (*input_dict)[TASK_DATA_KEY];
  }
};

IPIPE_REGISTER(Backend, Cache2OtherTensor);

class PrintTensor : public SingleBackend {
 public:
  void forward(dict input_dict) override {
    auto data = dict_gets<torch::Tensor>(input_dict, TASK_DATA_KEY);

    std::ostringstream oss;
    for (size_t i = 0; i < data.size(); ++i) {
      oss << "Tensor " << i << " " << data[i].sizes() << "\n";
    }

    for (const auto& item : data) {
      if (item.numel() > 60) {
        auto new_view = item.view(-1);                        // 将张量展平
        auto head = new_view.slice(0, 0, 5);                  // 取前5个元素
        auto tail = new_view.slice(0, -5, new_view.size(0));  // 取后5个元素
        oss << "Tensor is large. Shape: " << item.sizes() << ". Showing head and tail:\n";
        oss << head << "\n...\n" << tail << "\n";
      } else {
        oss << item << "\n";
      }
    }
    SPDLOG_WARN(oss.str());

    input_dict->operator[](TASK_RESULT_KEY) = (*input_dict)[TASK_DATA_KEY];
  }

 private:
  //   llamaRotaryEmbedding.forward(causal_mask, cache_position);
};

IPIPE_REGISTER(Backend, PrintTensor);
}  // namespace ipipe