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

#include <atomic>
#include <cassert>
#include <functional>
#include <memory>

#include <optional>
#include <unordered_map>
#include <variant>
#include <vector>

#include "omniback/core/any.hpp"
#include "omniback/core/string.hpp"
#include "omniback/helper/symbol.hpp"

namespace omniback {

#ifdef CUSTOM_DICT
using dict = CustomDict;
#else
using dict = std::shared_ptr<std::unordered_map<string, omniback::any>>;
inline dict make_dict() {
  return std::make_shared<std::unordered_map<string, omniback::any>>();
}

#endif

inline dict make_dict(string node_name, dict data = nullptr) {
  dict data_out;
  if (!data) {
    data_out = std::make_shared<std::unordered_map<string, any>>();
  } else {
    data_out = std::make_shared<std::unordered_map<string, any>>(*data);
  }
  assert(data_out != nullptr);
  if (!node_name.empty())
    (*data_out)["node_name"] = node_name;
  return data_out;
}

inline dict copy_dict(dict data) {
  dict data_out;
  if (!data) {
    data_out = std::make_shared<std::unordered_map<string, any>>();
  } else {
    data_out = std::make_shared<std::unordered_map<string, any>>(*data);
  }
  assert(data_out != nullptr);
  return data_out;
}

#define OMNI_NOEXCEPT noexcept
using dicts = std::vector<dict>;

// // stateful request
// struct Request {
//   uint32_t id;
//   uint32_t size;
//   // uint32_t slice_size;        // slice size: [offset, offset+size)
//   // uint32_t slice_offset = 0;  // slice offset
//   // std::unordered_map<string, string> custom;
// };

template <typename T = string>
T dict_get(dict data, const string& key, bool return_default = false) {
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

template <typename T = string>
T dict_pop(dict data, const string& key) {
  auto iter = data->find(key);
  if (iter != data->end()) {
    if (iter->second.type() != typeid(T))
      throw_wrong_type(typeid(T).name(), iter->second.type().name());
    T result = any_cast<T>(iter->second);
    data->erase(iter);
    return result;
  } else {
    throw std::invalid_argument("remove: can not found key: " + key);
  }
}

template <typename T = string>
std::vector<T> dict_gets(dict data, const string& key) {
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
  return std::vector<T>(); // make gcc happy
}

// template <typename T>
// std::optional<T> try_get(dict data, const string& key) {
//   auto iter = data->find(key);
//   if (iter != data->end()) {
//     if (iter->second.type() != typeid(T)) return std::nullopt;
//     T* result = any_cast<T>(&iter->second);
//     return *result;
//   } else {
//     throw_not_exist(key);
//   }
// }

template <typename T>
std::optional<T> try_get(dict data, const string& key) {
  auto iter = data->find(key);
  if (iter != data->end()) {
    if (iter->second.type() != typeid(T))
      return std::nullopt;
    T* result = any_cast<T>(&iter->second);
    return *result;
  }
  return std::nullopt;
}

template <typename T = string>
void update(dict data, const string& key, T& output) {
  auto iter = data->find(key);
  if (iter != data->end()) {
    output = any_cast<T>(iter->second);
  } else {
    throw std::invalid_argument("can not found key: " + key);
  }
}

template <typename T = string>
void try_update(dict data, const string& key, T& output) {
  auto iter = data->find(key);
  if (iter != data->end()) {
    output = any_cast<T>(iter->second);
  }
}

static inline void clear_dicts(dicts& data, const string& key) {
  for (auto& item : data) {
    item->erase(key);
  }
}

struct TypedDict {
  using BaseType = std::variant<
      bool,
      int32_t,
      std::shared_ptr<TypedDict>,
      string,
      double,
      std::vector<int32_t>,
      std::vector<
          std::string>>; // pls keep order for variant
                         // :https://github.com/pybind/pybind11/issues/1625

  std::unordered_map<string, BaseType> data;

  TypedDict(std::unordered_map<string, BaseType> data_in) : data(data_in) {}
  TypedDict() = default;
};

template <typename T>
T get(const TypedDict& data, const std::string& key) {
  auto iter = data.data.find(key);
  if (iter == data.data.end()) {
    throw std::runtime_error(key + " not found in TypedDict");
  }
  return std::get<T>(iter->second);
}

template <typename T>
T try_get(const TypedDict& data, const std::string& key) {
  auto iter = data.data.find(key);
  if (iter == data.data.end()) {
    return T();
  }
  return std::get<T>(iter->second);
}

template <>
inline bool get<bool>(const TypedDict& data, const std::string& key) {
  auto iter = data.data.find(key);
  if (iter == data.data.end()) {
    return false;
  }
  return std::get<bool>(iter->second);
}

template <typename T>
void try_update(const TypedDict& data, const std::string& key, T& result) {
  auto iter = data.data.find(key);
  if (iter != data.data.end()) {
    result = std::get<T>(iter->second);
  }
}

// template <typename T>
// struct ShareWrapper {
//   std::shared_ptr<T> ptr;
// };

} // namespace omniback