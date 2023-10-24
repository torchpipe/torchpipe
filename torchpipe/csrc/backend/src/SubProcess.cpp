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

#include "SubProcess.hpp"
#include "Sequential.hpp"

// #include <ATen/ATen.h>
#include "any.hpp"
#include "Backend.hpp"
#include "dict.hpp"
#include "dict_helper.hpp"
#include "params.hpp"
#include "reflect.h"
#include "time_utils.hpp"
#include "subprocess/subprocess.h"
#include "base_logging.hpp"
namespace ipipe {
bool SubProcess::init(const std::unordered_map<std::string, std::string>& config_param,
                      dict dict_config) {
  params_ = std::unique_ptr<Params>(new Params({{"SubProcess::backend", "Identity"}}, {}, {}, {}));

  if (!params_->init(config_param)) return false;

  engine_ = std::unique_ptr<Backend>(IPIPE_CREATE(Backend, params_->at("SubProcess::backend")));

  if (!engine_ || !engine_->init(config_param, dict_config)) {
    return false;
  }

  return true;
}

void SubProcess::forward(const std::vector<dict>& input_dicts) {
  // try {
  //   engine_->forward(input_dicts);
  // } catch (...) {
  //   cudaStreamSynchronize(c10::cuda::getCurrentCUDAStream());
  //   std::rethrow_exception(std::current_exception());
  // }
  engine_->forward(input_dicts);
}

IPIPE_REGISTER(Backend, SubProcess, "SubProcess");

}  // namespace ipipe