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

#include "BorrowReplay.hpp"

#include <chrono>
#include <functional>
#include <sstream>
#include <thread>
#include "base_logging.hpp"
#include "Backend.hpp"
#include "dict.hpp"
#include "time_utils.hpp"
#include "base_logging.hpp"
#include "dict_helper.hpp"
#include <ATen/ATen.h>

namespace ipipe {

BorrowReplay::~BorrowReplay() {}

bool BorrowReplay::init(const std::unordered_map<std::string, std::string>& config,
                       dict dict_config) {
  params_ = std::unique_ptr<Params>(new Params({}, {"max_batch_size"}, {}, {}));
  if (config.empty()) {
    SPDLOG_ERROR("empty config. Only support single-node configuration.");
    return false;
  }
  if (!params_->init(config)) return false;
  max_batch_size_ = std::stoi(params_->at("max_batch_size"));
  SPDLOG_INFO("max_batch_size: {}", max_batch_size_);
  return true;
}

void BorrowReplay::forward(const std::vector<dict>& raw_inputs) {
  // borrowed
  // borrow_or_insert set_replay get_replay borrow_all

  std::string borrow_type = dict_get<std::string>(raw_inputs[0], "borrow_type");
  if (borrow_type == "borrow_or_insert") {
    int id = dict_get<int>(raw_inputs[0], "id");
    auto input_tensor = dict_get<std::vector<at::Tensor>>(raw_inputs[0], TASK_DATA_KEY);

    IPIPE_ASSERT(input_tensor[0].size(0) < max_batch_size_);

    std::lock_guard<std::mutex> lock(lock_);
    if (pool_.size() + input_tensor[0].size(0) < max_batch_size_) {
      pool_.add(id, input_tensor);  // not wait here to get result
      return;
    } else {
      auto result = pool_.borrow(id, max_batch_size_ - input_tensor[0].size(0));
      (*raw_inputs[0])["result"] = result;
    }

  } else if (borrow_type == "borrow_all") {
    int id = dict_get<int>(raw_inputs[0], "id");
    auto result = pool_.borrow(id, pool_.size());
    (*raw_inputs[0])["result"] = result;

  } else if (borrow_type == "set_replay") {
    int id = dict_get<int>(raw_inputs[0], "id");
    auto input_tensor =
        dict_get<std::vector<std::vector<at::Tensor>>>(raw_inputs[0], TASK_DATA_KEY);
    pool_.set_replay(id, input_tensor);

  } else if (borrow_type == "get_replay") {
    int id = dict_get<int>(raw_inputs[0], "id");
    auto result = pool_.get_replay(id);
    (*raw_inputs[0])["result"] = result;
  } else if (borrow_type == "reset") {
    pool_.reset();
  } else {
    throw std::runtime_error(
        "borrow_type must be one of [borrow_or_insert, set_replay, "
        "get_replay, borrow_all, reset], but get " +
        borrow_type);
  }

  // if (raw_inputs[0]->at(TASK_DATA_KEY).type() == typeid(int)) {
  //   int id = dict_get<int>(raw_inputs[0], TASK_DATA_KEY);
  //   at::Tensor result = pool_.get(id);
  //   (*raw_inputs[0])["result"] = result;
  // }

  // auto input_tensor = dict_get<at::Tensor>(raw_inputs[0], TASK_DATA_KEY);
  // int id = dict_get<int>(raw_inputs[0], "id");
  // IPIPE_ASSERT(input_tensor.size(0) < max_batch_size_);
  // if (pool_.size() + input_tensor.size(0) < max_batch_size_) {
  //   pool_.add(id, input_tensor);  // not wait here to get result
  //   return;
  // }

  // borrowed successfully

  // const auto& input_shape = input_tensor.sizes().vec();

  return;
};
IPIPE_REGISTER(Backend, BorrowReplay, "BorrowReplay");

}  // namespace ipipe
