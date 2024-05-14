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

#include <gtest/gtest.h>

#include "Backend.hpp"
#include "dict.hpp"
#include "reflect.h"

#include <iostream>

#include <torch/torch.h>

TEST(cvtColorTensorTest, Basic) {
  auto* backend_instance = IPIPE_CREATE(ipipe::Backend, "cvtColorTensor");
  ASSERT_TRUE(backend_instance != nullptr);
  std::unordered_map<std::string, std::string> config;
  ASSERT_FALSE(backend_instance->init(config, nullptr));
  config["color"] = "rgb";
  ASSERT_TRUE(backend_instance->init(config, nullptr));

  auto input = torch::linspace(1, 27, 27).reshape({3, 3, 3});

  ipipe::dict data = ipipe::make_dict();
  (*data)[ipipe::TASK_DATA_KEY] = input;
  (*data)["color"] = std::string("bgr");

  backend_instance->forward({data});

  torch::Tensor result = ipipe::any_cast<torch::Tensor>(data->at(ipipe::TASK_RESULT_KEY));

  ASSERT_EQ((result[0].index({0, 0, 0}).item<float>()), 3.f);
  ASSERT_EQ((result[0].index({2, 0, 1}).item<float>()), 4.f);

  std::string color = ipipe::any_cast<std::string>(data->at("color"));
  ASSERT_EQ(color, "rgb");
}
