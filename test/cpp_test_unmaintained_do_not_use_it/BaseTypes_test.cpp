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
#include "Backend.hpp"

#include <iostream>
TEST(DictTest, BasicConstruction) {
  ipipe::dict a;
  ASSERT_EQ(a, nullptr);
  auto copy_dict = ipipe::make_dict("test_name", a);
  ASSERT_TRUE(copy_dict->find("node_name") != copy_dict->end() && copy_dict != a);
  auto new_dict = ipipe::make_dict("test_name");
  ASSERT_TRUE(new_dict->find("node_name") != new_dict->end() && new_dict != a);

  ASSERT_EQ("test_name", ipipe::dict_get(new_dict, "node_name", false));
  ASSERT_EQ("", ipipe::dict_get(new_dict, "not_exist_key_name", true));
}

TEST(BackendTest, BACKEND_CREATE) {
  std::unordered_map<std::string, std::string> config;

  auto all_names = IPIPE_ALL_NAMES(ipipe::Backend);
  for (const auto& item : all_names) std::cout << item << ",";
  std::cout << std::endl;

  EXPECT_LE(4, all_names.size());
  for (const auto backend : all_names) {
    auto* backend_instance = IPIPE_CREATE(ipipe::Backend, backend);
    ASSERT_TRUE(backend_instance != nullptr);
    if (backend_instance->init(config, nullptr)) {
      ASSERT_LE(backend_instance->min(), backend_instance->max());
      ASSERT_LE(1, backend_instance->min());
      ASSERT_LE(backend_instance->max(), UINT32_MAX);
    }
  }
}

TEST(BackendTest, BackendMax) {
  auto* backend_instance = IPIPE_CREATE(ipipe::Backend, "Max");
  ASSERT_TRUE(backend_instance != nullptr);
  std::unordered_map<std::string, std::string> config;
  ASSERT_TRUE(backend_instance->init(config, nullptr));

  config["Max::backend"] = "a,TASK_RESULT_KEY,c, d";
  ASSERT_FALSE(backend_instance->init(config, nullptr));
  config["Max::backend"] = "Range";
  config["max"] = "4";
  config["min"] = "2";
  ASSERT_FALSE(backend_instance->init(config, nullptr));
  config["Max::backend"] = "Range";
  config["max"] = "4";
  config["min"] = "1";
  ASSERT_TRUE(backend_instance->init(config, nullptr));
  ASSERT_EQ(backend_instance->max(), UINT32_MAX);
  ASSERT_EQ(backend_instance->min(), 1);
}
