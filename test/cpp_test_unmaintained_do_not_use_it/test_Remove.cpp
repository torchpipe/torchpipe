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

TEST(BackendTest, BackendRemoveV0) {
  auto* backend_instance = IPIPE_CREATE(ipipe::Backend, "RemoveV0");
  ASSERT_TRUE(backend_instance != nullptr);
  std::unordered_map<std::string, std::string> config;
  ASSERT_TRUE(backend_instance->init(config, nullptr));

  config["remove"] = "a,TASK_RESULT_KEY,c, d";
  ASSERT_FALSE(backend_instance->init(config, nullptr));
  config["remove"] = "a,TASK_DATA_KEY,c, d";
  ASSERT_FALSE(backend_instance->init(config, nullptr));

  config["remove"] = "a,c, d";
  ASSERT_TRUE(backend_instance->init(config, nullptr));
  ipipe::dict data = ipipe::make_dict("test_name");
  (*data)[ipipe::TASK_DATA_KEY] = int(-1);
  auto z = (*data)[ipipe::TASK_DATA_KEY];

  backend_instance->forward({data});
  ASSERT_EQ(data->size(), 3);

  ASSERT_EQ(data->at(ipipe::TASK_RESULT_KEY).type(), typeid(int));

  int result = ipipe::any_cast<int>(data->at(ipipe::TASK_RESULT_KEY));
  ASSERT_EQ(result, -1);

  ASSERT_TRUE(data->find("node_name") != data->end());
}