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

#include "SyncTensor.hpp"
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
namespace ipipe {
bool SyncTensor::init(const std::unordered_map<std::string, std::string>& config_param,
                      dict dict_config) {
  params_ = std::unique_ptr<Params>(new Params({{"_independent_thread_index", ""},
                                                {"priority", "high"},
                                                {"SyncTensor::backend", "Identity"}},
                                               {}, {}, {}));

  if (!params_->init(config_param)) return false;

  if (!params_->at("_independent_thread_index").empty()) {
    auto device_id_int = -1;  // std::stoi(device_id);
    int independent_thread_index = std::stoi(params_->at("_independent_thread_index"));
    assert(independent_thread_index >= 0);
    bool high_priority = ("high" == params_->at("priority"));
    high_priority = high_priority && (independent_thread_index < 32);

    bNeedSync_ = torch_not_use_default_stream(device_id_int, high_priority);
    SPDLOG_DEBUG("SyncTensor: sync enabled={} high_priority={}", bNeedSync_, high_priority);
  }

  if (config_param.find("device_id") != config_param.end()) {
    throw std::runtime_error(
        "SyncTensor: device_id is not supported by SyncTensor. Use `Torch` instead.");
  }

  c10::InferenceMode guard;  // optinal

  engine_ = std::unique_ptr<Backend>(IPIPE_CREATE(Backend, params_->at("SyncTensor::backend")));

  auto config_param_new = config_param;
  config_param_new.erase("SyncTensor::backend");

  if (!engine_ || !engine_->init(config_param_new, dict_config)) {
    return false;
  }

  c10::cuda::getCurrentCUDAStream().synchronize();

  return true;
}

void SyncTensor::forward(const std::vector<dict>& input_dicts) {
  c10::InferenceMode guard;  // optinal

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

IPIPE_REGISTER(Backend, SyncTensor, "SyncTensor");

}  // namespace ipipe