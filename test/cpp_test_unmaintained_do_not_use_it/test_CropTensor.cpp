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

TEST(CropTensorTest, Basic) {
  auto* backend_instance = IPIPE_CREATE(ipipe::Backend, "CropTensor");
  ASSERT_TRUE(backend_instance != nullptr);
  std::unordered_map<std::string, std::string> config;
  ASSERT_TRUE(backend_instance->init(config, nullptr));

  auto boxes = std::vector<std::vector<uint32_t>>{{1, 0, 11, 13}};
  auto input = at::zeros({1, 3, 100, 100});

  ipipe::dict data = ipipe::make_dict("test_name");
  (*data)[ipipe::TASK_DATA_KEY] = input;
  (*data)[ipipe::TASK_BOX_KEY] = boxes;

  backend_instance->forward({data});

  ASSERT_EQ(data->at(ipipe::TASK_RESULT_KEY).type(), typeid(std::vector<at::Tensor>));

  std::vector<at::Tensor> result =
      ipipe::any_cast<std::vector<at::Tensor>>(data->at(ipipe::TASK_RESULT_KEY));
  ASSERT_EQ(result.size(), 1);
  ASSERT_EQ(result[0].size(2), 13);
  ASSERT_EQ(result[0].size(3), 10);
}

TEST(CropTensorTest, BOXES) {
  auto* backend_instance = IPIPE_CREATE(ipipe::Backend, "CropTensor");
  ASSERT_TRUE(backend_instance != nullptr);
  std::unordered_map<std::string, std::string> config;
  ASSERT_TRUE(backend_instance->init(config, nullptr));

  auto boxes = std::vector<std::vector<uint32_t>>{{1, 0, 11, 13}, {1, 0, 11, 44}};
  auto input = at::zeros({100, 100, 4});

  ipipe::dict data = ipipe::make_dict();
  (*data)[ipipe::TASK_DATA_KEY] = input;
  (*data)[ipipe::TASK_BOX_KEY] = boxes;

  backend_instance->forward({data});

  std::vector<at::Tensor> result =
      ipipe::any_cast<std::vector<at::Tensor>>(data->at(ipipe::TASK_RESULT_KEY));

  ASSERT_EQ(result.size(), 2);
  ASSERT_TRUE(result[1].numel() > 0);
  ASSERT_EQ(result[1].size(-2), 44);
  ASSERT_EQ(result[1].size(-1), 10);
}

TEST(CropTensorTest, HW_INPUT) {
  auto* backend_instance = IPIPE_CREATE(ipipe::Backend, "CropTensor");
  ASSERT_TRUE(backend_instance != nullptr);
  std::unordered_map<std::string, std::string> config;
  ASSERT_TRUE(backend_instance->init(config, nullptr));

  auto boxes = std::vector<std::vector<uint32_t>>{{1, 0, 2, 13}, {1, 0, 11, 44}};
  auto input = at::zeros({1, 1, 100, 100});

  ipipe::dict data = ipipe::make_dict();
  (*data)[ipipe::TASK_DATA_KEY] = input;
  (*data)[ipipe::TASK_BOX_KEY] = boxes;

  backend_instance->forward({data});

  std::vector<at::Tensor> result =
      ipipe::any_cast<std::vector<at::Tensor>>(data->at(ipipe::TASK_RESULT_KEY));

  ASSERT_EQ(result.size(), 2);
  ASSERT_TRUE(result[1].numel() == 440);
  ASSERT_TRUE(result[1].sizes().size() == 4);

  ASSERT_EQ(result[1].size(-2), 44);
  ASSERT_EQ(result[1].size(-1), 10);
}
