// #include "backends/identity.hpp"
#include <algorithm>
#include <cmath>

#include "hami/builtin/basic_backends.hpp"
#include "hami/core/reflect.h"
#include "hami/helper/macro.h"
#include "hami/helper/base_logging.hpp"
#include "hami/core/task_keys.hpp"

namespace hami {
class Identity : public SingleBackend {
 public:
  void forward(const dict& input) override {
    auto iter = input->find(TASK_DATA_KEY);

    HAMI_ASSERT(iter != input->end(), "[`" + std::string(TASK_DATA_KEY) + "`] not found.");
    input->insert_or_assign(TASK_RESULT_KEY, iter->second);
  }
};
HAMI_REGISTER(Backend, Identity);

class Pow : public SingleBackend {
 public:
  enum class DataType { INT = 0, SIZE_T, FLOAT, DOUBLE, STRING };
  DataType data_type_{DataType::INT};
  void init(const std::unordered_map<std::string, std::string>& config,
            const dict& dict_config) override {
    auto iter = config.find("data_type");
    if (iter != config.end()) {
      const auto& data_type_str = iter->second;
      if (data_type_str == "INT") {
        data_type_ = DataType::INT;
      } else if (data_type_str == "SIZE_T") {
        data_type_ = DataType::SIZE_T;
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

    HAMI_ASSERT(iter != input->end(), "[`" + std::string(TASK_DATA_KEY) + "`] not found.");
    switch (data_type_) {
      case DataType::INT: {
        int data = any_cast<int>(iter->second);
        int result = static_cast<int>(std::pow(data, 2));
        // SPDLOG_INFO("Pow: {}", result);
        input->insert_or_assign(TASK_RESULT_KEY, result);
        break;
      }
      case DataType::SIZE_T: {
        size_t data = any_cast<size_t>(iter->second);
        size_t result = static_cast<size_t>(std::pow(data, 2));
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
HAMI_REGISTER(Backend, Pow);

class PrintKeys : public SingleBackend {
  void forward(const dict& input) override final {
    std::string keys;
    for (auto iter = input->begin(); iter != input->end(); ++iter) {
      keys += iter->first + " ";
    }
    SPDLOG_INFO("Keys: " + keys);
  };
};
HAMI_REGISTER(Backend, PrintKeys);

}  // namespace hami