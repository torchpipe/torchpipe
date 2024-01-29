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

#include "params.hpp"

#include "Backend.hpp"
#include "reflect.h"
#include "base_logging.hpp"
#include "Parallel.hpp"
#include <thread>
#include <future>

namespace ipipe {

bool Parallel::init(const std::unordered_map<std::string, std::string>& config, dict dict_config) {
  params_ =
      std::unique_ptr<Params>(new Params({{"Parallel::backend", "Identity"}, {}}, {}, {}, {}));

  if (!params_->init(config)) return false;

  backend_ = std::unique_ptr<Backend>(IPIPE_CREATE(Backend, params_->at("Parallel::backend")));
  if (backend_ && backend_->init(config, dict_config)) {
    backend_max_ = backend_->max();
    IPIPE_ASSERT(backend_max_ > 0);

    if (backend_->min() != 1) {
      SPDLOG_ERROR("backend_->min() != 1  min = {}", backend_->min());
      return false;
    }
    return true;
  }

  return false;
};

void Parallel::forward(const std::vector<dict>& input_dicts) {
  // std::async(std::launch::async, );
  if (input_dicts.size() <= backend_max_) {
    backend_->forward(input_dicts);
  } else {
    std::vector<std::future<void>> future_results;
    const std::size_t batch_num = input_dicts.size() / backend_max_;
    for (std::size_t i = 0; i < batch_num; ++i) {
      // backend_->forward();
      future_results.emplace_back(
          std::async(std::launch::async, &Backend::forward, backend_.get(),
                     std::vector<dict>(input_dicts.begin() + i * backend_max_,
                                       input_dicts.begin() + (i + 1) * backend_max_)));
    }
    int left_num = input_dicts.size() - batch_num * backend_max_;
    assert(left_num == 0 || left_num >= backend_->min());
    if (left_num > 0) {
      future_results.emplace_back(std::async(
          std::launch::async, &Backend::forward, backend_.get(),
          std::vector<dict>(input_dicts.begin() + batch_num * backend_max_, input_dicts.end())));
    }
    for (std::size_t i = 0; i < future_results.size(); ++i) {
      future_results[i].wait();
    }
  }
}

IPIPE_REGISTER(Backend, Parallel, "Parallel");
}  // namespace ipipe