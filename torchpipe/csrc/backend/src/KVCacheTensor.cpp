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
namespace ipipe {
bool KVCacheTensor::init(const std::unordered_map<std::string, std::string>& config_param,
                         dict dict_config) {
  params_ =
      std::unique_ptr<Params>(new Params({{"KVCacheTensor::backend", "Identity"}}, {}, {}, {}));

  if (!params_->init(config_param)) return false;

  engine_ = std::unique_ptr<Backend>(IPIPE_CREATE(Backend, params_->at("KVCacheTensor::backend")));

  auto config_param_new = config_param;
  config_param_new.erase("KVCacheTensor::backend");

  if (!engine_ || !engine_->init(config_param_new, dict_config)) {
    return false;
  }

  return true;
}

void KVCacheTensor::forward(dict input_dict) {
  auto iter = input_dict->find("request_id");
  if (iter != input_dict->end()) {
    std::string request_id = any_cast<std::string>(iter->second);
    SPDLOG_INFO("KVCacheTensor::forward request_id: {}", request_id);
  }
  engine_->forward({input_dict});
}

IPIPE_REGISTER(Backend, KVCacheTensor, "KVCacheTensor");

}  // namespace ipipe