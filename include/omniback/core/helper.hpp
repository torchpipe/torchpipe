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

#include <optional>
#include <vector>
#include "omniback/core/backend.hpp"
#include "omniback/core/dict.hpp"
#include "omniback/core/request_size.hpp"
#include "omniback/core/task_keys.hpp"
namespace omniback {

class Event;

/**
 * @brief Helper class for managing event lifecycle and ensuring proper
 * synchronization
 *
 * HasEventHelper manages event objects across multiple dictionaries, ensuring
 * consistency in asynchronous/synchronous states. It follows RAII principle to
 * guarantee event cleanup and provides synchronization capabilities.
 *
 * Usage:
 * - Create an HasEventHelper with a collection of dictionaries
 * - Call wait() to synchronize and clean up events
 * - Failing to call wait() before destruction will terminate the program
 *
 * Note: This class has no effect if all dictionaries already have events
 */
class HasEventHelper {
 public:
  HasEventHelper(const std::vector<dict>& data);
  void wait();
  ~HasEventHelper();

 private:
  const std::vector<dict>& dicts_;
  std::shared_ptr<Event> event_;
};

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
        if (item.second[i])
          (*dicts_[i])[item.first] = *item.second[i];
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
      throw std::out_of_range(
          "TASK_DATA_KEY is not allowed to be lazily erased");
    }
    lazy_clear_keys_.push_back(key);
    return *this;
  }
  DictHelper& keep(const std::string& key);

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

// make sure  the backend can handle  both sync && async
// situation even the user only implement the sync version
void event_guard_forward(
    std::function<void(const std::vector<dict>&)>,
    const std::vector<dict>& inputs);

void notify_event(const std::vector<dict>& io);

template <typename T>
std::string debug_node_info(const T& config) {
  if (auto iter = config.find(TASK_NODE_NAME_KEY); iter != config.end()) {
    return " node=" + iter->second + " . ";
  }

  return "";
}

std::string parse_dependency_from_param(
    const Backend* this_ptr,
    std::unordered_map<std::string, std::string>& config,
    std::string default_params_name,
    const std::string& default_dep_name = "");
std::optional<std::string> get_dependency_name(
    const Backend* this_ptr,
    const std::unordered_map<std::string, std::string>& config);
std::optional<std::string> get_dependency_name(
    const Backend* this_ptr,
    const std::unordered_map<std::string, std::string>& config,
    const std::string& default_cls_name);
std::string get_dependency_name(
    const Backend* this_ptr,
    const std::unordered_map<std::string, std::string>& config,
    const std::string& default_cls_name,
    const std::string& default_dep_name);
std::string get_dependency_name_force(
    const Backend* this_ptr,
    const std::unordered_map<std::string, std::string>& config);
std::string get_cls_name(
    const Backend* this_ptr,
    const std::string& default_cls_name);
namespace helper {
/**
 * Checks if all inputs have a specific key
 *
 * @param inputs Vector of dictionary items to check
 * @param key The key to check for
 * @return true if all items have the key
 * @return false if any item does not have the key
 */
inline bool all_has_key(
    const std::vector<dict>& inputs,
    const std::string& key) {
  return std::all_of(inputs.begin(), inputs.end(), [&key](const auto& item) {
    return item->find(key) != item->end();
  });
}

/**
 * Checks if none of the inputs have a specific key
 *
 * @param inputs Vector of dictionary items to check
 * @param key The key to check for
 * @return true if no items have the key
 * @return false if any item has the key
 */
inline bool none_has_key(
    const std::vector<dict>& inputs,
    const std::string& key) {
  return std::all_of(inputs.begin(), inputs.end(), [&key](const auto& item) {
    return item->find(key) == item->end();
  });
}

/**
 * Checks if all inputs have the key or none have the key, and the vector is not
 * empty
 *
 * @param inputs Vector of dictionary items to check
 * @param key The key to check for
 * @return true if all items have the key or none have the key, and the vector
 * is not empty
 * @return false otherwise
 */
inline bool none_or_all_has_key_and_unempty(
    const std::vector<dict>& inputs,
    const std::string& key) {
  if (inputs.empty()) {
    return false;
  }

  const bool has_key = inputs[0]->find(key) != inputs[0]->end();

  return std::all_of(
      inputs.begin() + 1, inputs.end(), [has_key, &key](const auto& item) {
        bool item_has_key = item->find(key) != item->end();
        return item_has_key == has_key;
      });
}
} // namespace helper
} // namespace omniback