// Copyright 2021-2023 NetEase.
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

#include "Torch.hpp"
#include "Sequential.hpp"

// #include <ATen/ATen.h>
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

namespace ipipe {
bool Torch::init(const std::unordered_map<std::string, std::string>& config_param,
                 dict dict_config) {
  params_ = std::unique_ptr<Params>(new Params(
      {{"_independent_thread_index", ""}, {"device_id", "-1"}}, {{"Torch::backend"}}, {}, {}));

  if (!params_->init(config_param)) return false;

  if (!params_->at("_independent_thread_index").empty()) {
    std::string device_id = params_->at("device_id");
    device_id_ = std::stoi(device_id);
    bNeedSync_ = torch_not_use_default_stream(device_id_, true);
    SPDLOG_DEBUG("Torch: bNeedSync_={} device_id={}", bNeedSync_, device_id_);
  }

  engine_ = std::unique_ptr<Backend>(IPIPE_CREATE(Backend, params_->at("Torch::backend")));

  auto config_param_new = config_param;
  config_param_new.erase("Torch::backend");

  if (!engine_ || !engine_->init(config_param_new, dict_config)) {
    return false;
  }

  return true;
}

void Torch::forward(const std::vector<dict>& input_dicts) {
  if (device_id_ != -1) {
    for (auto& input : input_dicts) {
      auto& input_data = *input;
      // 从any里取出at::Tensor 或者 std::vector<at::Tensor> ,将其的device更改为device_id_
      if (input_data.at(TASK_DATA_KEY).type() == typeid(at::Tensor)) {
        at::Tensor data = any_cast<at::Tensor>(input_data.at(TASK_DATA_KEY));
        data = to_current_device(data);
        input_data[TASK_DATA_KEY] = data;
      } else if (input_data.at(TASK_DATA_KEY).type() == typeid(std::vector<at::Tensor>)) {
        std::vector<at::Tensor> datas =
            any_cast<std::vector<at::Tensor>>(input_data.at(TASK_DATA_KEY));
        std::vector<at::Tensor> result;
        for (auto data : datas) {
          result.emplace_back(to_current_device(data));
        }
        c10::cuda::getCurrentCUDAStream().synchronize();
        input_data[TASK_DATA_KEY] = result;
      }
    }
  }

  if (bNeedSync_) {
    try {
      engine_->forward(input_dicts);
    } catch (...) {
      c10::cuda::getCurrentCUDAStream().synchronize();
      std::rethrow_exception(std::current_exception());
    }
    c10::cuda::getCurrentCUDAStream().synchronize();
  } else {
    engine_->forward(input_dicts);
  }
}

IPIPE_REGISTER(Backend, Torch, "Torch");

}  // namespace ipipe