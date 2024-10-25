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

#pragma once

#include <atomic>
#include <cassert>
#include <condition_variable>
#include <functional>
#include <iostream>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>
#include <optional>

#include "any.hpp"

namespace ipipe {
// using namespace nonstd;  // for any, any_cast

class CustomDict {
 public:
  explicit CustomDict(std::unordered_map<std::string, ipipe::any>* ptr = nullptr) : data_(ptr) {}
  explicit CustomDict(std::shared_ptr<std::unordered_map<std::string, ipipe::any>> ptr)
      : data_(ptr) {}
  ~CustomDict() = default;

  // Smart pointer operations
  void reset(std::unordered_map<std::string, ipipe::any>* ptr = nullptr) { data_.reset(ptr); }

  std::unordered_map<std::string, ipipe::any>* get() const { return data_.get(); }

  void swap(CustomDict& other) noexcept { data_.swap(other.data_); }

  operator bool() const { return static_cast<bool>(data_); }

  std::unordered_map<std::string, ipipe::any>& operator*() { return *data_; }
  const std::unordered_map<std::string, ipipe::any>& operator*() const { return *data_; }
  std::unordered_map<std::string, ipipe::any>* operator->() { return data_.get(); }
  const std::unordered_map<std::string, ipipe::any>* operator->() const { return data_.get(); }

  // Implicit conversion to std::shared_ptr
  operator std::shared_ptr<std::unordered_map<std::string, ipipe::any>>() const { return data_; }

 private:
  std::shared_ptr<std::unordered_map<std::string, ipipe::any>> data_;
};

#ifdef CUSTOM_DICT
using dict = CustomDict;
#else
using dict = std::shared_ptr<std::unordered_map<std::string, ipipe::any>>;
#endif

static inline dict make_dict(std::string node_name = "", dict data = nullptr) {
  dict data_out;
  if (!data) {
    data_out = std::make_shared<std::unordered_map<std::string, any>>();
  } else {
    data_out = std::make_shared<std::unordered_map<std::string, any>>(*data);
  }
  assert(data_out != nullptr);
  if (!node_name.empty()) (*data_out)["node_name"] = node_name;
  return data_out;
}

#define IPIPE_NOEXCEPT noexcept
using dicts = std::vector<dict>;
using mapmap = std::unordered_map<std::string, std::unordered_map<std::string, std::string>>;

/**
 * @if  chinese
 * @brief 代表输入数据的键值
 * @else
 * @endif
 */
constexpr auto TASK_DATA_KEY = "data";
/**
 * @if  chinese
 * @brief 代表输出数据的键值。
 * 如果一个后端处理完的数据无此键值，调用方可认为出现异常。
 * @else
 * @endif
 */
constexpr auto TASK_RESULT_KEY = "result";
constexpr auto TASK_CPU_RESULT_KEY = "cpu_result";

constexpr auto TASK_CONTEXT_KEY = "context";

/*! */
constexpr auto TASK_BOX_KEY = "_box";
/*! type: std::unordered_map<std::string, std::string> */
constexpr auto TASK_INFO_KEY = "_info";

constexpr auto TASK_STACK_KEY = "_stack";

constexpr auto TASK_NODE_NAME_KEY = "node_name";

constexpr auto TASK_DEFAULT_NAME_KEY = "_default_node_name";
constexpr auto TASK_REQUEST_KEY = "request";
constexpr auto TASK_REQUEST_SIZE_KEY = "request_size";

constexpr auto TASK_RESTART_KEY = "restart";

static inline const std::unordered_map<std::string, std::string> TASK_KEY_MAP(
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

static inline void try_replace_inner_key(std::string& key) {
  static const std::string prefix = "TASK_";
  static const std::string suffix = "_KEY";
  static const size_t prefix_suffix_len = prefix.size() + suffix.size();

  if (key.size() >= prefix_suffix_len && key.compare(0, prefix.size(), prefix) == 0 &&
      key.compare(key.size() - suffix.size(), suffix.size(), suffix) == 0) {
    const auto iter = TASK_KEY_MAP.find(key);
    if (iter == TASK_KEY_MAP.end()) {
      throw std::runtime_error("Inner key not supported: " + key);
    }
    key = iter->second;
  }
  return;
}

static inline bool is_reserved(const std::string& key) {
  static const std::unordered_set<std::string> reserved_words{"global", "default", "node_name", ""};
  if (0 != reserved_words.count(key)) return true;
  for (const auto& item : TASK_KEY_MAP) {
    if (item.second == key) return true;
  }
  return false;
}
// stateful request
struct Request {
  uint32_t id;
  uint32_t size;
  // uint32_t slice_size;        // slice size: [offset, offset+size)
  // uint32_t slice_offset = 0;  // slice offset
  // std::unordered_map<std::string, std::string> custom;
};

extern void throw_wrong_type(const char* need_type, const char* input_type);
extern void throw_not_exist(const std::string& key);
template <typename T = std::string>
T dict_get(dict data, const std::string& key, bool return_default = false) {
  auto iter = data->find(key);
  if (iter != data->end()) {
    if (iter->second.type() != typeid(T))
      throw_wrong_type(typeid(T).name(), iter->second.type().name());
    T result = any_cast<T>(iter->second);
    return result;
  } else {
    if (return_default)
      return T();
    else {
      throw std::invalid_argument("dict_get: can not found key: " + key);
    }
  }
}

template <typename T = std::string>
std::vector<T> dict_gets(dict data, const std::string& key) {
  auto iter = data->find(key);
  if (iter != data->end()) {
    if (iter->second.type() == typeid(T)) {
      T* result = any_cast<T>(&iter->second);
      return std::vector<T>{*result};
    } else if (iter->second.type() == typeid(std::vector<T>)) {
      return *any_cast<std::vector<T>>(&iter->second);
    } else {
      throw_wrong_type(typeid(T).name(), iter->second.type().name());
    }
  } else {
    throw std::invalid_argument("dict_get: can not found key: " + key);
  }
  return std::vector<T>();  // make gcc happy
}

template <typename T>
std::optional<T> try_get(dict data, const std::string& key) {
  auto iter = data->find(key);
  if (iter != data->end()) {
    if (iter->second.type() != typeid(T)) return std::nullopt;
    T* result = any_cast<T>(&iter->second);
    return *result;
  } else {
    throw_not_exist(key);
  }
}

template <typename Container = std::vector<dict>>
static inline int get_request_size(const Container& in) {
  int total_len = 0;
  for (const auto& item : in) {
    const auto iter = item->find(TASK_REQUEST_SIZE_KEY);
    if (iter != item->end()) {
      const int re = any_cast<int>(iter->second);
      total_len += re;
    } else {
      total_len += 1;
    }
  }
  return total_len;
}

template <>
inline int get_request_size<dict>(const dict& in) {
  const auto iter = in->find(TASK_REQUEST_SIZE_KEY);
  if (iter != in->end()) {
    return any_cast<int>(iter->second);
  } else {
    return 1;
  }
}

template <typename T = std::string>
void update(dict data, const std::string& key, T& output) {
  auto iter = data->find(key);
  if (iter != data->end()) {
    output = any_cast<T>(iter->second);
  } else {
    throw std::invalid_argument("can not found key: " + key);
  }
}

template <typename T = std::string>
void try_update(dict data, const std::string& key, T& output) {
  auto iter = data->find(key);
  if (iter != data->end()) {
    output = any_cast<T>(iter->second);
  }
}

static inline void clear_dicts(dicts& data, const std::string& key) {
  for (auto& item : data) {
    item->erase(key);
  }
}

}  // namespace ipipe