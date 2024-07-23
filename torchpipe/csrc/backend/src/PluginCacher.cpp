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

#include "resource_guard.hpp"
#include "c10/cuda/CUDAStream.h"
#include "PluginCacher.hpp"
#include "reflect.h"
#include "ipipe_common.hpp"
// #include <torch/torch.h>

namespace ipipe {

bool PluginCacher::init(const std::unordered_map<std::string, std::string>& config_param,
                        dict dict_config) {
  params_ = std::unique_ptr<Params>(new Params({}, {"PluginCacher::backend"}, {}, {}));

  if (!params_->init(config_param)) return false;

  IPIPE_ASSERT(dict_config);
  pack_config_param_ = BackendConfig{.config = config_param, .dict_config = dict_config};
  config_.add(pack_config_param_, (void*)c10::cuda::getCurrentCUDAStream().stream());

  engine_ = std::unique_ptr<Backend>(IPIPE_CREATE(Backend, params_->at("PluginCacher::backend")));

  auto config_param_new = config_param;
  config_param_new.erase("PluginCacher::backend");

  if (!engine_ || !engine_->init(config_param_new, dict_config)) {
    return false;
  }

  return true;
}

void PluginCacher::forward(const std::vector<dict>& input_dicts) {
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  ForwardGuard guard(input_dicts, (void*)stream);
  engine_->forward(input_dicts);
}

const std::vector<dict>& PluginCacher::query_input() {
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  const auto* re = ForwardGuard<std::vector<dict>>::query_input((void*)stream);
  IPIPE_ASSERT(re != nullptr);
  return *re;
}

const std::vector<dict>& PluginCacher::query_input(void* stream) {
  const auto* re = ForwardGuard<std::vector<dict>>::query_input((void*)stream);
  IPIPE_ASSERT(re != nullptr);
  return *re;
}
const BackendConfig& PluginCacher::query_config(void* stream) {
  const auto* re = PluginCacher::config_.query((void*)stream);
  IPIPE_ASSERT(re != nullptr);
  return *re;
}

ResourceGuard<BackendConfig> PluginCacher::config_;

IPIPE_REGISTER(Backend, PluginCacher, "PluginCacher");

}  // namespace ipipe