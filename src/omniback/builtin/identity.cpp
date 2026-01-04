// #include "backends/identity.hpp"
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
// #include
namespace omniback {
class Identity : public BackendOne {
 public:
  void forward(const dict& io) override {
    auto iter = io->find(TASK_DATA_KEY);

    OMNI_ASSERT(
        iter != io->end(), "[`" + std::string(TASK_DATA_KEY) + "`] not found.");
    io->insert_or_assign(TASK_RESULT_KEY, iter->second);
  }
};

class AsU64Identity : public BackendOne {
 public:
  void forward(const dict& io) override {
    auto iter = io->find(TASK_DATA_KEY);

    OMNI_ASSERT(
        iter != io->end(), "[`" + std::string(TASK_DATA_KEY) + "`] not found.");
    auto data = iter->second.cast<uint64_t>();
    io->insert_or_assign(TASK_RESULT_KEY, data);
  }
};

OMNI_REGISTER(Backend, Identity);
OMNI_REGISTER(Backend, AsU64Identity);

class Pow : public BackendOne {
 public:
  enum class DataType { INT = 0, SIZE_T, FLOAT, DOUBLE, STRING };
  DataType data_type_{DataType::INT};
  void impl_init(
      const std::unordered_map<std::string, std::string>& config,
      const dict& kwargs) override {
    auto iter = config.find("data_type");
    if (iter != config.end()) {
      const auto& data_type_str = iter->second;
      if (data_type_str == "INT") {
        data_type_ = DataType::INT;
      }
      // else if (data_type_str == "SIZE_T")
      // {
      //     data_type_ = DataType::SIZE_T;
      // }
      else if (data_type_str == "FLOAT") {
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

    OMNI_ASSERT(
        iter != input->end(),
        "[`" + std::string(TASK_DATA_KEY) + "`] not found.");
    switch (data_type_) {
      case DataType::INT: {
        int data = any_cast<int>(iter->second);
        int result = static_cast<int>(std::pow(data, 2));
        // SPDLOG_INFO("Pow: {}", result);
        input->insert_or_assign(TASK_RESULT_KEY, result);
        break;
      }
      // case DataType::SIZE_T:
      // {
      //     size_t data = any_cast<size_t>(iter->second);
      //     size_t result = static_cast<size_t>(std::pow(data, 2));
      //     input->insert_or_assign(TASK_RESULT_KEY, result);
      //     break;
      // }
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

class PrintKeys : public BackendOne {
  void forward(const dict& input) override final {
    std::string keys;
    for (auto iter = input->begin(); iter != input->end(); ++iter) {
      keys += iter->first + " ";
    }
    SPDLOG_INFO("Keys: " + keys);
  };
};
OMNI_REGISTER(Backend, PrintKeys);

class OMNI_EXPORT Identities : public Backend {
 private:
  void impl_init(
      const std::unordered_map<std::string, std::string>& config,
      const dict& kwargs) override final {
    max_ = str::get<size_t>(config, "max");
  }
  void impl_forward(const std::vector<dict>& input_output) override final {
    for (const auto& item : input_output) {
      (*item)[TASK_RESULT_KEY] = item->at(TASK_DATA_KEY);
    }
  }
  [[nodiscard]] uint32_t impl_max() const override final {
    return max_;
  }

 private:
  uint32_t max_{std::numeric_limits<uint32_t>::max()};
};
OMNI_REGISTER_BACKEND(Identities);

class OMNI_EXPORT TimeStamp : public Backend {
 private:
  void impl_init(
      const std::unordered_map<std::string, std::string>& params,
      const dict& options) override final {
    auto args_kwargs = parser_v2::get_args_kwargs(this, "TimeStamp", params);
    OMNI_ASSERT(
        args_kwargs.first.size() == 1,
        "Requires exactly ==1 argument. Usage: TimeStamp(key)/TimeStamp::args=key_to_time");
    key_ = args_kwargs.first[0];
  }
  void impl_forward(const std::vector<dict>& input_output) override final {
    float time = get_time();
    for (const auto& item : input_output) {
      (*item)[TASK_RESULT_KEY] = item->at(TASK_DATA_KEY);
      (*item)[key_] = time;
    }
  }
  [[nodiscard]] uint32_t impl_max() const override final {
    return max_;
  }

 private:
  float get_time() {
    return static_cast<float>(
        std::chrono::duration<double>(
            std::chrono::system_clock::now().time_since_epoch())
            .count());
  }
  uint32_t max_{std::numeric_limits<uint32_t>::max()};
  std::string key_;
};
OMNI_REGISTER_BACKEND(TimeStamp);

class OMNI_EXPORT LogTime : public Backend {
 private:
  void impl_init(
      const std::unordered_map<std::string, std::string>& params,
      const dict& options) override final {
    auto args_kwargs = parser_v2::get_args_kwargs(this, "LogTime", params);
    OMNI_ASSERT(
        args_kwargs.first.size() == 1,
        "Requires exactly ==1 argument. Usage: LogTime(key)/LogTime::args=key_to_time");
    key_ = args_kwargs.first[0];
  }
  void impl_forward(const std::vector<dict>& input_output) override final {
    // float time = get_time();
    float time = helper::timestamp();
    SPDLOG_INFO("timer: {} = {}", key_, time);
    for (const auto& item : input_output) {
      (*item)[TASK_RESULT_KEY] = item->at(TASK_DATA_KEY);
    }
  }
  [[nodiscard]] uint32_t impl_max() const override final {
    return max_;
  }

 private:
  float get_time() {
    return static_cast<float>(
        std::chrono::duration<double>(
            std::chrono::system_clock::now().time_since_epoch())
            .count());
  }
  uint32_t max_{std::numeric_limits<uint32_t>::max()};
  std::string key_;
};
OMNI_REGISTER_BACKEND(LogTime);
} // namespace omniback