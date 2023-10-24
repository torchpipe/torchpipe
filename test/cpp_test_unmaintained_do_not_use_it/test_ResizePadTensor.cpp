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

#include <gtest/gtest.h>

#include "Backend.hpp"
#include "dict.hpp"
#include "reflect.h"

#include <iostream>

#include <ATen/ATen.h>

TEST(ResizePadTensorTest, Basic) {
  auto* backend_instance = IPIPE_CREATE(ipipe::Backend, "ResizePadTensor");
  ASSERT_TRUE(backend_instance != nullptr);
  std::unordered_map<std::string, std::string> config;
  ASSERT_FALSE(backend_instance->init(config, nullptr));
  config["max_h"] = "224";
  config["max_w"] = "12";
  ASSERT_TRUE(backend_instance->init(config, nullptr));

  auto input = at::linspace(1, 3 * 42 * 27, 3 * 42 * 27).reshape({42, 27, 3});

  ipipe::dict data = ipipe::make_dict();
  (*data)[ipipe::TASK_DATA_KEY] = input;

  backend_instance->forward({data});

  at::Tensor result = ipipe::any_cast<at::Tensor>(data->at(ipipe::TASK_RESULT_KEY));
  ASSERT_EQ(result.size(2), 224);
  ASSERT_EQ(result.size(3), 12);
  ASSERT_EQ(result.size(1), 3);

  input = input.cpu();
  (*data)[ipipe::TASK_DATA_KEY] = input;
  backend_instance->forward({data});

  result = ipipe::any_cast<at::Tensor>(data->at(ipipe::TASK_RESULT_KEY));
  ASSERT_EQ(result.size(2), 224);
  ASSERT_EQ(result.size(3), 12);
  ASSERT_EQ(result.size(1), 3);
}

TEST(ResizePadTensorTest, NCHW) {
  auto* backend_instance = IPIPE_CREATE(ipipe::Backend, "ResizePadTensor");
  std::unordered_map<std::string, std::string> config;
  config["max_h"] = "444";
  config["max_w"] = "224";
  ASSERT_TRUE(backend_instance->init(config, nullptr));

  ipipe::dict data = ipipe::make_dict();

  // hw4
  auto input = at::linspace(1, 4 * 42 * 27, 4 * 42 * 27).reshape({42, 27, 4});
  (*data)[ipipe::TASK_DATA_KEY] = input;
  data->erase(ipipe::TASK_RESULT_KEY);
  backend_instance->forward({data});
  ASSERT_NE(data->find(ipipe::TASK_RESULT_KEY), data->end());

  // 3hw
  input = at::linspace(1, 3 * 42 * 27, 3 * 42 * 27).reshape({3, 42, 27});
  (*data)[ipipe::TASK_DATA_KEY] = input;
  data->erase(ipipe::TASK_RESULT_KEY);
  ASSERT_THROW(backend_instance->forward({data}), std::out_of_range);
  // ASSERT_EQ(data->find(ipipe::TASK_RESULT_KEY), data->end());

  // nhw3
  input = at::linspace(1, 4 * 42 * 27, 4 * 42 * 27).reshape({1, 42, 27, 4});
  (*data)[ipipe::TASK_DATA_KEY] = input;
  data->erase(ipipe::TASK_RESULT_KEY);
  ASSERT_THROW(backend_instance->forward({data}), std::out_of_range);
  // ASSERT_NE(data->find(ipipe::TASK_RESULT_KEY), data->end());
}
