// Copyright 2021-2026 NetEase.
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

#include <algorithm>
#include <cmath>

#include "omniback/builtin/basic_backends.hpp"
#include "omniback/core/helper.hpp"
#include "omniback/core/reflect.h"
#include "omniback/core/task_keys.hpp"
#include "omniback/helper/base_logging.hpp"
#include "omniback/helper/macro.h"
#include "omniback/helper/string.hpp"
#include "omniback/helper/timer.hpp"

namespace om {
/**
 * @brief Identity backend - passes data through unchanged
 */
class Identity : public BackendOne {
 public:
  void forward(const dict& io) override {
    auto iter = io->find(TASK_DATA_KEY);
    OMNI_ASSERT(iter != io->end(), 
                "[`" + std::string(TASK_DATA_KEY) + "`] not found.");
    io->insert_or_assign(TASK_RESULT_KEY, iter->second);
  }
};

/**
 * @brief Identity backend that casts data to uint64_t
 */
class AsU64Identity : public BackendOne {
 public:
  void forward(const dict& io) override {
    auto iter = io->find(TASK_DATA_KEY);
    OMNI_ASSERT(iter != io->end(),
                "[`" + std::string(TASK_DATA_KEY) + "`] not found.");
    auto data = iter->second.cast<uint64_t>();
    io->insert_or_assign(TASK_RESULT_KEY, data);
  }
};

OMNI_REGISTER(Backend, Identity);
OMNI_REGISTER(Backend, AsU64Identity);

/**
 * @brief Pow backend - squares input values
 */
class Pow : public BackendOne {
 public:
  enum class DataType { INT = 0, FLOAT, DOUBLE, STRING };
  
 private:
  DataType data_type_{DataType::INT};
  
 public:
  void impl_init(
      const std::unordered_map<std::string, std::string>& config,
      const dict& kwargs) override {
    auto iter = config.find("data_type");
    if (iter != config.end()) {
      const auto& data_type_str = iter->second;
      if (data_type_str == "INT") {
        data_type_ = DataType::INT;
      } else if (data_type_str == "FLOAT") {
        data_type_ = DataType::FLOAT;
      } else if (data_type_str == "DOUBLE") {
        data_type_ = DataType::DOUBLE;
      } else if (data_type_str == "STRING") {
        data_type_ = DataType::STRING;
      }
    }
  }
  
  void forward(const dict& input) override {
    auto iter = input->find(TASK_DATA_KEY);
    OMNI_ASSERT(iter != input->end(),
                "[`" + std::string(TASK_DATA_KEY) + "`] not found.");
    
    switch (data_type_) {
      case DataType::INT: {
        int data = any_cast<int>(iter->second);
        int result = static_cast<int>(std::pow(data, 2));
        input->insert_or_assign(TASK_RESULT_KEY, result);
        break;
      }
      case DataType::FLOAT: {
        float data = any_cast<float>(iter->second);
        float result = std::pow(data, 2);
        input->insert_or_assign(TASK_RESULT_KEY, result);
        break;
      }
      case DataType::DOUBLE: {
        double data = any_cast<double>(iter->second);
        double result = std::pow(data, 2);
        input->insert_or_assign(TASK_RESULT_KEY, result);
        break;
      }
      case DataType::STRING: {
        std::string data = any_cast<std::string>(iter->second);
        std::string result = std::to_string(std::pow(std::stod(data), 2));
        input->insert_or_assign(TASK_RESULT_KEY, result);
        break;
      }
      default:
        throw std::runtime_error("[Pow] data type not supported");
    }
  }
};
OMNI_REGISTER(Backend, Pow);

/**
 * @brief PrintKeys backend - logs all keys in the input dict
 */
class PrintKeys : public BackendOne {
  void forward(const dict& input) override final {
    std::string keys;
    for (const auto& [key, _] : *input) {
      keys += key + " ";
    }
    SPDLOG_INFO("Keys: {}", keys);
  }
};
OMNI_REGISTER(Backend, PrintKeys);

/**
 * @brief Identities backend - batch identity operation
 */
class OMNI_EXPORT Identities : public Backend {
 private:
  uint32_t max_{std::numeric_limits<uint32_t>::max()};

  void impl_init(
      const std::unordered_map<std::string, std::string>& config,
      const dict& kwargs) override final {
    max_ = static_cast<uint32_t>(str::get<size_t>(config, "max"));
  }
  
  void impl_forward(const std::vector<dict>& input_output) override final {
    for (const auto& item : input_output) {
      (*item)[TASK_RESULT_KEY] = item->at(TASK_DATA_KEY);
    }
  }
  
  [[nodiscard]] uint32_t impl_max() const override final {
    return max_;
  }
};
OMNI_REGISTER_BACKEND(Identities);

/**
 * @brief TimeStamp backend - adds timestamp to each output
 */
class OMNI_EXPORT TimeStamp : public Backend {
 private:
  uint32_t max_{std::numeric_limits<uint32_t>::max()};
  std::string key_;

  void impl_init(
      const std::unordered_map<std::string, std::string>& params,
      const dict& options) override final {
    auto args_kwargs = parser_v2::get_args_kwargs(this, "TimeStamp", params);
    OMNI_ASSERT(args_kwargs.first.size() == 1,
                "Requires exactly 1 argument. Usage: TimeStamp(key)");
    key_ = args_kwargs.first[0];
  }
  
  void impl_forward(const std::vector<dict>& input_output) override final {
    float time = helper::timestamp();
    for (const auto& item : input_output) {
      (*item)[TASK_RESULT_KEY] = item->at(TASK_DATA_KEY);
      (*item)[key_] = time;
    }
  }
  
  [[nodiscard]] uint32_t impl_max() const override final {
    return max_;
  }
};
OMNI_REGISTER_BACKEND(TimeStamp);

/**
 * @brief LogTime backend - logs current timestamp
 */
class OMNI_EXPORT LogTime : public Backend {
 private:
  uint32_t max_{std::numeric_limits<uint32_t>::max()};
  std::string key_;

  void impl_init(
      const std::unordered_map<std::string, std::string>& params,
      const dict& options) override final {
    auto args_kwargs = parser_v2::get_args_kwargs(this, "LogTime", params);
    OMNI_ASSERT(args_kwargs.first.size() == 1,
                "Requires exactly 1 argument. Usage: LogTime(key)");
    key_ = args_kwargs.first[0];
  }
  
  void impl_forward(const std::vector<dict>& input_output) override final {
    float time = helper::timestamp();
    SPDLOG_INFO("timer: {} = {}", key_, time);
    for (const auto& item : input_output) {
      (*item)[TASK_RESULT_KEY] = item->at(TASK_DATA_KEY);
    }
  }
  
  [[nodiscard]] uint32_t impl_max() const override final {
    return max_;
  }
};
OMNI_REGISTER_BACKEND(LogTime);
} // namespace om