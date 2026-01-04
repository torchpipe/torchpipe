// Copyright 2021-2025 NetEase.
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

#pragma once

#include <exception>
#include <unordered_map>
#include <unordered_set>
// #include  "omniback/core/dict.hpp"
#include <stdexcept>
#include "omniback/core/string.hpp"

namespace omniback {

constexpr auto TASK_DATA_KEY = "data";
constexpr auto TASK_RESULT_KEY = "result";

constexpr auto TASK_OUTPUT_KEY = "output";

constexpr auto TASK_TMP_KEY = "tmp";
constexpr auto TASK_STATUS_KEY = "status";
constexpr auto TASK_MSG_KEY = "MSG";

constexpr auto TASK_CPU_RESULT_KEY = "cpu_result";

constexpr auto TASK_CONTEXT_KEY = "context";

constexpr auto TASK_BOX_KEY = "BOX";
constexpr auto TASK_INFO_KEY = "INFO";

constexpr auto TASK_STACK_KEY = "STACK";

constexpr auto TASK_NODE_NAME_KEY = "node_name";
constexpr auto TASK_ENTRY_KEY = "entrypoint";

constexpr auto TASK_DEFAULT_NAME_KEY = "default_node_name";
constexpr auto TASK_INDEX_KEY = "_independent_index";

constexpr auto TASK_RESTART_KEY = "restart";
constexpr auto TASK_REQUEST_ID_KEY = "request_id";
constexpr auto TASK_WAITING_EVENT_KEY = "waiting_event";

using id_type = std::string;

constexpr auto TASK_DEFAULT_NODE_NAME_KEY = "default_node_name";
constexpr auto TASK_GLOBAL_KEY = "global";
constexpr auto TASK_CONFIG_KEY = "config";
constexpr auto TASK_NEXT_KEY = "next";
constexpr auto TASK_MAP_KEY = "map";
constexpr auto TASK_OR_KEY = "or";
constexpr auto TASK_STREAM_KEY = "stream";

static inline const std::unordered_map<string, string> TASK_KEY_MAP(
    {{"TASK_RESULT_KEY", TASK_RESULT_KEY},
     {"TASK_DATA_KEY", TASK_DATA_KEY},
     {"TASK_BOX_KEY", TASK_BOX_KEY},
     {"TASK_INFO_KEY", TASK_INFO_KEY},
     {"TASK_NODE_NAME_KEY", TASK_NODE_NAME_KEY},
     {"TASK_CONTEXT_KEY", TASK_CONTEXT_KEY},
     {"TASK_RESTART_KEY", TASK_RESTART_KEY},
     {"TASK_STACK_KEY", TASK_STACK_KEY},
     {"TASK_DEFAULT_NAME_KEY", TASK_DEFAULT_NAME_KEY}});

// template <typename T>
// bool try_replace_inner_key(T& key) {
//   return false;
// }

bool try_replace_inner_key(std::string& key);

template <typename T>
inline void try_replace_inner_key(std::unordered_map<string, T>& kv) {
  std::unordered_map<string, T> re;
  for (const auto& [key, value] : kv) {
    string new_key = key;
    T new_value = value;

    try_replace_inner_key(new_key);
    if constexpr (std::is_same_v<T, std::string>) {
      try_replace_inner_key(new_value);
    }
    re[new_key] = new_value;
  }
  std::swap(kv, re);
}

static inline bool is_reserved(const string& key) {
  static const std::unordered_set<string> reserved_words{
      "global", "default", "node_name", ""};
  if (0 != reserved_words.count(key))
    return true;
  for (const auto& item : TASK_KEY_MAP) {
    if (item.second == key)
      return true;
  }
  return false;
}

} // namespace omniback