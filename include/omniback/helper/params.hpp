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

#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace omniback {

/**
 * @brief Manages parameters for each Backend. It maintains two types of
 * parameters:
 * - init_optional_params: Optional parameters during initialization.
 * - init_required_params: Required parameters during initialization.
 */
class Params {
 public:
  Params() = default;

  Params(
      std::unordered_map<std::string, std::string> init_optional_params,
      std::set<std::string> init_required_params = {})
      : init_optional_params_(std::move(init_optional_params)),
        init_required_params_(std::move(init_required_params)) {}

  /**
   * @brief Initialization function. It mainly:
   * - Saves optional parameters (init_optional_params) to the internal
   * parameter _config (which aggregates all required parameters and their
   * values). If the input parameter config has updated values, it saves the
   * updated values to _config.
   * - Checks if config contains all required parameters
   * (init_required_params) and saves them to _config. If not, it prints an
   * error and returns false, indicating initialization failure.
   *
   * @param config Key-value pairs of parameters.
   */
  void impl_init(const std::unordered_map<std::string, std::string>& config);

  std::string& at(const std::string& key);

  std::string& operator[](const std::string& key) {
    return config_[key];
  }

  /**
   * @brief Inserts a key-value pair into _config.
   * @param key The key of the parameter.
   * @param value The value of the parameter.
   */
  void set(const std::string& key, const std::string& value) {
    config_[key] = value;
  }

 private:
  std::unordered_map<std::string, std::string> config_;

  std::unordered_map<std::string, std::string> init_optional_params_;
  std::set<std::string> init_required_params_;
};

} // namespace omniback