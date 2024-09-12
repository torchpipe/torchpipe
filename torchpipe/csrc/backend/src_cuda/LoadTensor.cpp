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

/**
 * @brief 用于加载磁盘中的tensor(.pt文件)到文件，可加载 python中  ``torch.save()`` 保存的文件 .
 * 如果想加载为图片，可连续使用
 * @ref LoadTensor , @ref Tensor2Mat.
 * @warning 一般用做调试；
 */
class LoadTensor : public SingleBackend {
 public:
  /**
   * @param tensor_name 文件路径；
   *
   */
  virtual bool init(const std::unordered_map<std::string, std::string>& config_param,
                    dict) override {
    params_ = std::unique_ptr<Params>(new Params({}, {"tensor_name"}, {}, {}));
    if (!params_->init(config_param)) return false;

    std::ifstream file(params_->at("tensor_name").c_str());
    if (!file.good()) {
      SPDLOG_ERROR("LoadTensor: dir " + params_->at("tensor_name") + " not exists.");
      throw std::invalid_argument("dir " + params_->at("tensor_name") + " not exists.");
    }
    return true;
  }

  /**
   * @param TASK_RESULT_KEY 加载的tensor
   */
  virtual void forward(dict input_dict) override {
    const std::string& save_name = params_->at("tensor_name");
    if (!save_name.empty()) {
      // imwrite(save_name, input_tensor);
      std::ifstream file(params_->at("tensor_name").c_str());
      if (!file.good()) {
        SPDLOG_ERROR("LoadTensor: dir " + params_->at("tensor_name") + " not exists.");
        throw std::invalid_argument("dir " + params_->at("tensor_name") + " not exists.");
      }
      file.seekg(0, file.end);
      int length = file.tellg();
      file.seekg(0, file.beg);

      std::vector<char> data(length);
      file.read(data.data(), length);

      auto data_loaded = torch::pickle_load(data).toTensor();
      (*input_dict)[TASK_RESULT_KEY] = data_loaded;
    }
  }

 private:
  std::unique_ptr<Params> params_;
};

IPIPE_REGISTER(Backend, LoadTensor, "LoadTensor");

class EmbedTokensTensor : public SingleBackend {
 public:
  /**
   * @param tensor_name 文件路径；
   *
   */
  virtual bool init(const std::unordered_map<std::string, std::string>& config_param,
                    dict) override {
    params_ = std::unique_ptr<Params>(new Params({}, {"embed_tokens"}, {}, {}));
    if (!params_->init(config_param)) return false;

    SPDLOG_INFO("load " + params_->at("embed_tokens"));
    std::ifstream file(params_->at("embed_tokens").c_str());
    if (!file.good()) {
      throw std::invalid_argument(params_->at("embed_tokens") + " not exists.");
    }
    file.seekg(0, file.end);
    int length = file.tellg();
    file.seekg(0, file.beg);

    std::vector<char> data(length);
    file.read(data.data(), length);

    tensor_ = torch::pickle_load(data).toTensor().cuda();

    return true;
  }

  /**
   * @param TASK_RESULT_KEY 加载的tensor
   */
  virtual void forward(dict input_dict) override {
    torch::Tensor input = any_cast<torch::Tensor>(input_dict->at(TASK_DATA_KEY));
    // slice   tensor from input
    if (input.is_cpu()) {
      input = input.cuda();
    }
    bool need_batch = false;
    if (input.dim() == 2) {
      need_batch = true;
      IPIPE_ASSERT(input.size(0) == 1);
      input = input.squeeze(0);
    }
    torch::Tensor data_loaded = tensor_.index_select(0, input);
    if (need_batch) data_loaded = data_loaded.unsqueeze(0);
    (*input_dict)[TASK_RESULT_KEY] = data_loaded;
  }

 private:
  std::unique_ptr<Params> params_;
  torch::Tensor tensor_;
};

IPIPE_REGISTER(Backend, EmbedTokensTensor, "EmbedTokensTensor");

}  // namespace ipipe
