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
        if (item.second[i].has_value()) (*dicts_[i])[item.first] = item.second[i];
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
    std::vector<any> keeped;
    for (const auto& da : dicts_) {
      auto iter = da->find(key);
      if (iter == da->end()) {
        // throw std::out_of_range(key + ": not exists");
        keeped.emplace_back(any());
      } else
        keeped.emplace_back(iter->second);
    }
    keep_[key] = keeped;
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
  std::unordered_map<std::string, std::vector<any>> keep_;
  std::unordered_map<std::string, std::vector<any>> keep_alive_;
};
}  // namespace ipipe
