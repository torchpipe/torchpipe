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

#include <unordered_map>
#include <exception>
#include <unordered_set>
// #include  "hami/core/dict.hpp"
#include "hami/core/string.hpp"
#include <stdexcept>

namespace hami {

constexpr auto TASK_DATA_KEY = "data";
constexpr auto TASK_RESULT_KEY = "result";
constexpr auto TASK_TMP_KEY = "tmp";

constexpr auto TASK_CPU_RESULT_KEY = "cpu_result";

constexpr auto TASK_CONTEXT_KEY = "context";

constexpr auto TASK_BOX_KEY = "_box";
constexpr auto TASK_INFO_KEY = "_info";

constexpr auto TASK_STACK_KEY = "_stack";

constexpr auto TASK_NODE_NAME_KEY = "node_name";
constexpr auto TASK_ENTRY_KEY = "entrypoint";

constexpr auto TASK_DEFAULT_NAME_KEY = "default_node_name";
constexpr auto TASK_INDEX_KEY = "_independent_index";

constexpr auto TASK_REQUEST_KEY = "request";
constexpr auto TASK_REQUEST_SIZE_KEY = "request_size";

constexpr auto TASK_RESTART_KEY = "restart";
constexpr auto TASK_REQUEST_ID_KEY = "request_id";

constexpr auto TASK_DEFAULT_NODE_NAME_KEY = "default_node_name";
constexpr auto TASK_GLOBAL_KEY = "global";
constexpr auto TASK_CONFIG_KEY = "config";
constexpr auto TASK_NEXT_KEY = "next";
constexpr auto TASK_MAP_KEY = "map";
constexpr auto TASK_OR_KEY = "or";

static inline const std::unordered_map<string, string> TASK_KEY_MAP(
    {{"TASK_RESULT_KEY", TASK_RESULT_KEY},
     {"TASK_DATA_KEY", TASK_DATA_KEY},
     {"TASK_BOX_KEY", TASK_BOX_KEY},
     {"TASK_INFO_KEY", TASK_INFO_KEY},
     {"TASK_NODE_NAME_KEY", TASK_NODE_NAME_KEY},
     {"TASK_CONTEXT_KEY", TASK_CONTEXT_KEY},
     {"TASK_REQUEST_KEY", TASK_REQUEST_KEY},
     {"TASK_RESTART_KEY", TASK_RESTART_KEY},
     {"TASK_STACK_KEY", TASK_STACK_KEY},
     {"TASK_DEFAULT_NAME_KEY", TASK_DEFAULT_NAME_KEY},
     {"TASK_REQUEST_SIZE_KEY", TASK_REQUEST_SIZE_KEY}});

static inline void try_replace_inner_key(string& key) {
    static const string prefix = "TASK_";
    static const string suffix = "_KEY";
    static const size_t prefix_suffix_len = prefix.size() + suffix.size();

    if (key.size() >= prefix_suffix_len &&
        key.compare(0, prefix.size(), prefix) == 0 &&
        key.compare(key.size() - suffix.size(), suffix.size(), suffix) == 0) {
        const auto iter = TASK_KEY_MAP.find(key);
        if (iter == TASK_KEY_MAP.end()) {
            throw std::runtime_error("Inner key not supported: " + key);
        }
        key = iter->second;
    }
    return;
}

static inline bool is_reserved(const string& key) {
    static const std::unordered_set<string> reserved_words{"global", "default",
                                                           "node_name", ""};
    if (0 != reserved_words.count(key)) return true;
    for (const auto& item : TASK_KEY_MAP) {
        if (item.second == key) return true;
    }
    return false;
}

}  // namespace hami