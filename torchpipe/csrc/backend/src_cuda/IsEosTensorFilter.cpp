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
#include <fstream>
#include "base_logging.hpp"
#include "reflect.h"
#include "dict.hpp"
#include "params.hpp"

#include "filter.hpp"

namespace ipipe {

class IsEosTensorFilter : public SingleBackend {
 public:
  virtual bool init(const std::unordered_map<std::string, std::string>& config_param,
                    dict) override {
    params_ = std::unique_ptr<Params>(new Params({{"eos", "1"}}, {}, {}, {}));
    if (!params_->init(config_param)) return false;

    eos_ = std::stoi(params_->at("eos"));

    return true;
  }

  virtual void forward(dict input_dict) override {
    int now_token{-1};
    auto& input = *input_dict;
    if (input[TASK_DATA_KEY].type() == typeid(torch::Tensor)) {
      auto* input_tensor = any_cast<torch::Tensor>(&input[TASK_DATA_KEY]);
      now_token = input_tensor->item<long>();
    } else {
      SPDLOG_ERROR("IsEosTensorFilter: torch::Tensor needed; error input type: " +
                   std::string(input[TASK_DATA_KEY].type().name()));
      throw std::runtime_error("IsEosTensorFilter: torch::Tensor needed; error input type: " +
                               std::string(input[TASK_DATA_KEY].type().name()));
    }
    // input[TASK_RESULT_KEY] = input[TASK_DATA_KEY];
    if (now_token == eos_) {
      input["filter"] = Filter::status::Run;
    } else {
      input["filter"] = Filter::status::Skip;
    }
    // input[TASK_RESULT_KEY] = input[TASK_DATA_KEY];
  }

 private:
  std::unique_ptr<Params> params_;
  int eos_{-100};
};

IPIPE_REGISTER(Backend, IsEosTensorFilter, "IsEosTensorFilter");

}  // namespace ipipe