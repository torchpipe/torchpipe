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

#include <torch/serialize.h>
#include <torch/torch.h>
#include <fstream>
#include "base_logging.hpp"
#include "reflect.h"
#include "dict.hpp"
#include "params.hpp"

namespace {
inline const std::string thread_id_string() {
  std::stringstream ss;
  ss << std::this_thread::get_id();
  return ss.str();
}
};  // namespace
namespace ipipe {

class MultiModalEmbedsTensor : public SingleBackend {
 public:
  /**
   * @param tensor_name 文件路径；
   *
   */
  virtual bool init(const std::unordered_map<std::string, std::string>& config_param,
                    dict shared_dict) override {
    params_ = std::unique_ptr<Params>(new Params({{"input_embeds", ""}}, {}, {}, {}));
    if (!params_->init(config_param)) return false;

    auto save_name = params_->at("input_embeds");
    if (save_name.empty()) {
      SPDLOG_INFO("MultiModalEmbedsTensor: input_embeds is empty. Skip loading.");
      return true;
    }
    if (!save_name.empty()) {
      read(save_name, tensors_);
      SPDLOG_INFO("MultiModalEmbedsTensor: load input embeds from " + save_name);
    }
    // IPIPE_ASSERT(tensorstensors.size() == 2);
    return true;
  }

  void read(const std::string& save_name, std::vector<torch::Tensor> tensors) {
    std::ifstream file(save_name);
    if (!file.good()) {
      SPDLOG_ERROR("MultiModalEmbedsTensor: dir " + save_name + " not exists.");
      throw std::invalid_argument("dir " + save_name + " not exists.");
    }
    file.seekg(0, file.end);
    int length = file.tellg();
    file.seekg(0, file.beg);

    std::vector<char> data(length);
    file.read(data.data(), length);

    auto data_loaded = torch::pickle_load(data);

    if (data_loaded.isTensor()) {
      const torch::Tensor input = data_loaded.toTensor();
      throw std::runtime_error(
          "MultiModalEmbedsTensor: input is a tensor, but we need a list of tensors.");
    } else if (data_loaded.isTensorList()) {
      tensors = data_loaded.toTensorVector();  // c10::List
    } else if (data_loaded.isTuple()) {
      const auto& ivalue_vector = data_loaded.toTupleRef().elements().vec();
      for (const auto& ivalue : ivalue_vector) {
        if (ivalue.isTensor()) {
          tensors.push_back(ivalue.toTensor());
        } else {
          // Handle the case where the IValue is not a Tensor
          throw std::runtime_error(
              "MultiModalEmbedsTensor from tuple: input is not a list of tensors.");
        }
      }
    } else if (data_loaded.isList()) {
      const auto& ivalue_vector = data_loaded.toList();
      for (const auto& item : ivalue_vector) {
        const auto& ivalue = item.get();
        if (ivalue.isTensor()) {
          tensors.push_back(ivalue.toTensor());
        } else {
          // Handle the case where the IValue is not a Tensor
          throw std::runtime_error(
              "MultiModalEmbedsTensor from list: input is not a list of tensors.");
        }
      }
    } else {
      SPDLOG_ERROR("MultiModalEmbedsTensor: input is not a list of tensors.");
      throw std::runtime_error("MultiModalEmbedsTensor: input is not a list of tensors: ");
    }

    std::stringstream ss;

    ss << "Load tensor(s) from " << save_name << ". Shape: ";
    for (auto& item : tensors) {
      if (item.sizes().size() == 2) {
        item = item.unsqueeze(0);
      }
      ss << item.sizes() << " ";
      item = item.cuda();
    }
    SPDLOG_INFO(ss.str());
  }

  /**
   * @param TASK_RESULT_KEY 加载的tensor
   */
  virtual void forward(dict input_dict) override {
    std::vector<torch::Tensor> image_embeds;
    auto& input = *input_dict;

    if (input[TASK_DATA_KEY].type() == typeid(torch::Tensor)) {
      torch::Tensor input_tensor = any_cast<torch::Tensor>(input[TASK_DATA_KEY]);

      image_embeds.push_back(input_tensor);
    } else if (input[TASK_DATA_KEY].type() == typeid(std::vector<torch::Tensor>)) {
      image_embeds = any_cast<std::vector<torch::Tensor>>(input[TASK_DATA_KEY]);

    } else {
      throw std::runtime_error(
          "MultiModalEmbedsTensor: input is not a tensor or a list of tensors.");
    }
    for (auto& input_tensor : image_embeds) {
      if (!input_tensor.is_cuda()) {
        input_tensor = input_tensor.to(torch::kCUDA);
        IPIPE_ASSERT(input_tensor.sizes().size() == 3);
      }
    }
    std::vector<torch::Tensor> results;
    if (tensors.size() == image_embeds.size()) {
      for (std::size_t i = 0; i < tensors.size(); ++i) {
        results.push_back(tensors[i]);
        results.push_back(image_embeds[i]);
      }
    } else if (tensors.size() == image_embeds.size() + 1) {
      for (std::size_t i = 0; i < image_embeds.size(); ++i) {
        results.push_back(tensors[i]);
        results.push_back(image_embeds[i]);
      }
      results.push_back(tensors.back());
    } else {
      throw std::runtime_error(
          "MultiModalEmbedsTensor: number of image_embeds and llm_embeds  not match.");
    }
    torch::Tensor re = torch::cat(results, 1);
    (*input_dict)[TASK_RESULT_KEY] = re;
  }

 private:
  std::unique_ptr<Params> params_;
  std::vector<torch::Tensor> tensors_;
};

IPIPE_REGISTER(Backend, MultiModalEmbedsTensor, "MultiModalEmbedsTensor");

class Append : public SingleBackend {
 public:
  /**
   * @param tensor_name 文件路径；
   *
   */
  virtual bool init(const std::unordered_map<std::string, std::string>& config_param,
                    dict shared_dict) override {}

  virtual void forward(dict input_dict) override {};
};
IPIPE_REGISTER(Backend, MultiModalEmbedsTensor, "MultiModalEmbedsTensor");

}  // namespace ipipe