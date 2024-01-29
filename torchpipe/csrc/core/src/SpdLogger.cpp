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
#include "reflect.h"
#include "Backend.hpp"

namespace ipipe {

std::shared_ptr<spdlog::logger> default_logger() { return spdlog::default_logger(); }
spdlog::logger* default_logger_raw() { return spdlog::default_logger_raw(); }
/**
 * @brief 配置
 * spdlog::default_logger，可以在系统初始化时调用。
 * @remarks 此后端被 Interpreter 设置为 Interpreter::env 的默认值。
 * Interpreter 将在初始化开始时，初始化Interpreter::env后端。
 * @todo 考虑 Interpreter级别的日志替换进程级别的日志。
 */
class SpdLogger : public EmptyForwardSingleBackend {
 public:
  /**
   * @brief
   * @param SpdLogger::pattern 设置 spdlog::default_logger
   * 的打印模式。 参考 https://spdlog.docsforge.com/v1.x/3.custom-formatting/
   * @param SpdLogger::path 指定日志文件。如果不指定，将直接打印到控制台。
   *
   */
  virtual bool init(const std::unordered_map<std::string, std::string>& config,
                    dict dict_config) override final {
    auto iter_logger = config.find("SpdLogger::path");
    if (iter_logger != config.end()) {
      const auto& log_path = iter_logger->second;
      if (!log_path.empty()) {
        spdlog::default_logger()->sinks().clear();
        spdlog::default_logger()->sinks().push_back(
            std::make_shared<spdlog::sinks::basic_file_sink_mt>(log_path));
      }
    }

    // logger
    iter_logger = config.find("SpdLogger::pattern");
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
    SPDLOG_INFO("SpdLogger loaded.");

    return true;
  };
};

IPIPE_REGISTER(Backend, SpdLogger, "SpdLogger");

std::string colored(const std::string& message) {
  return fmt::format(fmt::bg(fmt::terminal_color::cyan) | fmt::fg(fmt::terminal_color::black) |
                         fmt::emphasis::bold,
                     message);
}

std::string colored(const char* message) {
  return fmt::format(fmt::bg(fmt::terminal_color::cyan) | fmt::fg(fmt::terminal_color::black) |
                         fmt::emphasis::bold,
                     message);
}

}  // namespace ipipe