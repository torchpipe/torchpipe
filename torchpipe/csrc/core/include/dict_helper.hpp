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

#include "dict.hpp"
#include <sstream>
namespace ipipe {
class DictHelper {
 public:
  DictHelper(const std::vector<dict>& data)
      : dicts_(data) {
          // keep(TASK_DATA_KEY);
          // keep("node_name");
        };
  ~DictHelper() {
    for (std::size_t i = 0; i < dicts_.size(); ++i) {
      for (const std::string& key : lazy_clear_keys_) {
        dicts_[i]->erase(key);
      }
      for (const auto& item : keep_) {
        if (item.second[i]) (*dicts_[i])[item.first] = *item.second[i];
      }
    }
  };
  // void clear(const std::string& key) {
  //   for (auto da : dicts_) {
  //     da->erase(key);
  //   }
  // }
  DictHelper& erase(const std::string& key) {
    for (auto da : dicts_) {
      da->erase(key);
    }
    return *this;
  }
  template <typename T>
  DictHelper& set(const std::string& key, T&& value) {
    for (auto da : dicts_) {
      (*da)[key] = value;
    }
    return *this;
  }

  DictHelper& lazy_erase(const std::string& key) {
    if (key == TASK_DATA_KEY) {
      throw std::out_of_range("TASK_DATA_KEY is not allowed to be lazily erased");
    }
    lazy_clear_keys_.push_back(key);
    return *this;
  }
  DictHelper& keep(const std::string& key) {
    std::vector<std::optional<any>> keeped;
    for (const auto& da : dicts_) {
      auto iter = da->find(key);
      if (iter == da->end()) {
        keeped.emplace_back(std::nullopt);
      } else {
        keeped.emplace_back(iter->second);
      }
    }
    keep_[key] = keeped;
    return *this;
  }

  DictHelper& lazy_copy(const std::string& key, const std::string& target_key) {
    std::vector<std::optional<any>> keeped;
    for (const auto& da : dicts_) {
      auto iter = da->find(key);
      if (iter == da->end()) {
        keeped.emplace_back(std::nullopt);
      } else {
        keeped.emplace_back(iter->second);
      }
    }
    keep_[target_key] = keeped;
    return *this;
  }

  DictHelper& copy(const std::string& key, const std::string& target_key) {
    std::vector<std::optional<any>> keeped;
    for (auto& da : dicts_) {
      auto iter = da->find(key);
      if (iter == da->end()) {
      } else {
        (*da)[target_key] = iter->second;
      }
    }
    return *this;
  }

  DictHelper& keep_alive(const std::string& key) {
    std::vector<any> keeped;
    for (const auto& da : dicts_) {
      auto iter = da->find(key);
      if (iter == da->end()) {
        // throw std::out_of_range(key + ": not exists");
      }
      keeped.emplace_back(iter->second);
    }
    keep_alive_[key] = keeped;
    return *this;
  }

 private:
  const std::vector<dict>& dicts_;
  std::vector<std::string> lazy_clear_keys_;
  std::unordered_map<std::string, std::vector<std::optional<any>>> keep_;
  std::unordered_map<std::string, std::vector<any>> keep_alive_;
};

// 独立的 Helper 类，不与 TypedDict 耦合
class TypedDictHelper {
 public:
  // 现在需要接受一个 TypedDict 的引用作为参数
  template <typename T>
  static T get(TypedDict& idict, const std::string& key) {
    auto it = idict.data.find(key);
    if (it == idict.data.end()) {
      throw std::out_of_range(key + ": not exists");
    }

    if constexpr (std::is_same_v<T, TypedDict>) {
      auto ptr = std::get<std::shared_ptr<TypedDict>>(it->second);
      return ptr;
    } else {
      return std::get<T>(it->second);
    }
  }

  template <typename T>
  static void set(TypedDict& idict, const std::string& key, const T& value) {
    if constexpr (std::is_same_v<T, TypedDict>) {
      idict.data[key] = std::make_shared<TypedDict>(value);
    } else {
      idict.data[key] = value;
    }
  }

  static bool contains(TypedDict& idict, const std::string& key) {
    return idict.data.count(key) > 0;
  }

  static void remove(TypedDict& idict, const std::string& key) { idict.data.erase(key); }

  static std::vector<std::string> keys(TypedDict& dict) {
    std::vector<std::string> result;
    for (const auto& pair : dict.data) {
      result.push_back(pair.first);
    }
    return result;
  }

  static std::string get_repr(const TypedDict& self, size_t depth = 0) {
    constexpr size_t max_length = 50;
    constexpr size_t max_depth = 20;

    if (depth > max_depth) {
      return "{...}";
    }

    std::string repr = "{";
    for (const auto& pair : self.data) {
      repr += "\"" + pair.first + "\": ";

      repr += std::visit(
          [depth](auto&& arg) -> std::string {
            using T = std::decay_t<decltype(arg)>;
            if constexpr (std::is_same_v<T, std::shared_ptr<TypedDict>>) {
              return get_repr(*arg, depth + 1);
            } else if constexpr (std::is_same_v<T, std::string>) {
              if (arg.length() > max_length) {
                return "\"" + arg.substr(0, max_length / 2) + " ... " +
                       arg.substr(arg.length() - max_length / 2) + "\"";
              } else {
                return "\"" + arg + "\"";
              }
            } else if constexpr (std::is_same_v<T, bool>) {
              return std::string(arg ? "true" : "false");
            } else if constexpr (std::is_arithmetic_v<T>) {
              std::ostringstream out_stream;
              out_stream << arg;
              return out_stream.str();
            }

            std::ostringstream out_stream;
            out_stream << arg;
            return out_stream.str();
            // return std::to_string(arg);
          },
          pair.second);
      repr += ", ";
    }
    repr = repr.substr(0, repr.length() - 2);  // 去掉最后的", "
    repr += "}";
    return repr;
  }
};
}  // namespace ipipe
