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

#include "base_logging.hpp"
#include <spdlog/fmt/bundled/color.h>

#include "spdlog/sinks/stdout_sinks.h"
#include <spdlog/sinks/basic_file_sink.h>
#include "spdlog/sinks/rotating_file_sink.h"
#include "spdlog/sinks/stdout_color_sinks.h"
#include "reflect.h"
#include "Backend.hpp"
#include <atomic>

#include <mutex>
namespace ipipe {

class FileLogger : public EmptyForwardSingleBackend {
 public:
  virtual bool init(const std::unordered_map<std::string, std::string>& config,
                    dict dict_config) override final {
    // 读取FileLogger::maxBytes
    auto iter_maxByte = config.find("FileLogger::maxBytes");
    if (iter_maxByte != config.end()) {
      auto config_maxByte = std::stoi(iter_maxByte->second);
      if (config_maxByte > 0 && config_maxByte < 1024 * 1024 * 1024) {
        this->maxBytes = config_maxByte;
      } else {
        SPDLOG_ERROR("Logger config error, please check: maxByte > 0 and maxByte < 1G ");
        return false;
      }
    }

    // 读取FileLogger::backupCount
    auto iter_backupCount = config.find("FileLogger::backupCount");
    if (iter_backupCount != config.end()) {
      auto config_backupCount = std::stoi(iter_backupCount->second);
      if (config_backupCount >= 0 && config_backupCount < 1024) {
        this->backupCount = config_backupCount;
      } else {
        SPDLOG_ERROR("Logger config error, please check: backupCount >= 0 and backupCount < 1024 ");
        return false;
      }
    }

    // 读取FileLogger::path
    auto iter_logger = config.find("FileLogger::path");
    if (iter_logger != config.end()) {
      const auto& log_path = iter_logger->second;
      if (!log_path.empty()) {
        // -1为maxBytes默认值，不等于-1，说明配置maxByte参数，将采用rotate方式保存
        if (maxBytes != -1) {
          // rotate print to file
          spdlog::default_logger()->sinks().clear();
          spdlog::default_logger()->sinks().push_back(
              std::make_shared<spdlog::sinks::rotating_file_sink_mt>(log_path, maxBytes,
                                                                     backupCount));
        } else {
          // not rotate print to file
          spdlog::default_logger()->sinks().clear();
          spdlog::default_logger()->sinks().push_back(
              std::make_shared<spdlog::sinks::basic_file_sink_mt>(log_path));
        }

      } else {
        SPDLOG_ERROR("FileLogger config error, FileLogger::path not exist");
        return false;
      }
    } else {
      SPDLOG_ERROR("FileLogger config error, must have FileLogger::path");
      return false;
    }

    // logger
    iter_logger = config.find("FileLogger::pattern");
    if (iter_logger != config.end()) {
      spdlog::default_logger()->set_pattern(iter_logger->second);
    } else {
#ifndef NDEBUG
      spdlog::default_logger()->set_pattern("[%L][%m/%d %H:%M:%S][%s:%# %!]: %v");
#else
      spdlog::default_logger()->set_pattern("[%l][%m/%d %H:%M:%S]: %v");
#endif
    }

#ifndef NDEBUG
    spdlog::default_logger()->set_level(spdlog::level::debug);
#else
    spdlog::default_logger()->set_level(spdlog::level::info);
#endif
    SPDLOG_INFO("FileLogger loaded.");

    return true;
  };

 private:
  int maxBytes = -1;  // -1 代表默认，不设置循环保存策略
  int backupCount = 0;
};
IPIPE_REGISTER(Backend, FileLogger, "FileLogger");

}  // namespace ipipe