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

class LoadTensors : public SingleBackend {
 public:
  /**
   * @param tensor_name 文件路径；
   *
   */
  virtual bool init(const std::unordered_map<std::string, std::string>& config_param,
                    dict shared_dict) override {
    params_ = std::unique_ptr<Params>(new Params({}, {"tensor_name"}, {}, {}));
    if (!params_->init(config_param)) return false;

    read();

    return true;
  }

  void read() {
    const std::string& save_name = params_->at("tensor_name");
    if (!save_name.empty()) {
      std::ifstream file(params_->at("tensor_name").c_str());
      if (!file.good()) {
        SPDLOG_ERROR("LoadTensors: dir " + params_->at("tensor_name") + " not exists.");
        throw std::invalid_argument("dir " + params_->at("tensor_name") + " not exists.");
      }
      file.seekg(0, file.end);
      int length = file.tellg();
      file.seekg(0, file.beg);

      std::vector<char> data(length);
      file.read(data.data(), length);

      auto data_loaded = torch::pickle_load(data);

      if (data_loaded.isTensor()) {
        const torch::Tensor input = data_loaded.toTensor();
        throw std::runtime_error("LoadTensors: input is a tensor, but we need a list of tensors.");
      } else if (data_loaded.isTensorList()) {
        tensors_ = data_loaded.toTensorVector();  // c10::List
      } else if (data_loaded.isTuple()) {
        const auto& ivalue_vector = data_loaded.toTupleRef().elements().vec();
        for (const auto& ivalue : ivalue_vector) {
          if (ivalue.isTensor()) {
            tensors_.push_back(ivalue.toTensor());
          } else {
            // Handle the case where the IValue is not a Tensor
            throw std::runtime_error("LoadTensors from tuple: input is not a list of tensors.");
          }
        }
      } else if (data_loaded.isList()) {
        const auto& ivalue_vector = data_loaded.toList();
        for (const auto& item : ivalue_vector) {
          const auto& ivalue = item.get();
          if (ivalue.isTensor()) {
            tensors_.push_back(ivalue.toTensor());
          } else {
            // Handle the case where the IValue is not a Tensor
            throw std::runtime_error("LoadTensors from list: input is not a list of tensors.");
          }
        }
      } else {
        SPDLOG_ERROR("LoadTensors: input is not a list of tensors.");
        throw std::runtime_error("LoadTensors: input is not a list of tensors: ");
      }
    }
    std::stringstream ss;

    ss << "Load tensor(s) from " << save_name << ". Shape: ";
    for (const auto& item : tensors_) {
      ss << item.sizes() << " ";
    }
    SPDLOG_INFO(ss.str());
  }

  /**
   * @param TASK_RESULT_KEY 加载的tensor
   */
  virtual void forward(dict input_dict) override { (*input_dict)[TASK_RESULT_KEY] = tensors_; }

 private:
  std::unique_ptr<Params> params_;
  std::vector<torch::Tensor> tensors_;
};

IPIPE_REGISTER(Backend, LoadTensors, "LoadTensors");

}  // namespace ipipe