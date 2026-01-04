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
#include <string>

#ifndef SPDLOG_ACTIVE_LEVEL
#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_DEBUG
#endif

#include "spdlog/spdlog.h"

namespace omniback {

// namespace spdlog {
// class logger;
// }
std::shared_ptr<spdlog::logger> default_logger(); // cross dynamic libraries.
spdlog::logger* default_logger_raw();
std::string colored(const std::string& message);

namespace {

class LoggerGuard {
 public:
  LoggerGuard() {
    std::lock_guard<std::mutex> lock(lock_);
    auto in_default = default_logger();
    auto now_logger = spdlog::default_logger();
    if (in_default != now_logger && in_default)
      spdlog::set_default_logger(in_default);
  };

 private:
  static std::mutex lock_;
};
std::mutex LoggerGuard::lock_;
static LoggerGuard g_tmp_lock_guard;

} // namespace
} // namespace omniback

// enum class omniback_log_level { trace = 0, debug, info, warn, err, critical,
// off, n_levels }; void print_logger(const char*,int,const char*,); #define
// OMNI_LOGGER_INFO(file, line, func, ...) print_logger(file, line, func,
// __VA_ARGS__) #define LOG_INFO(...) OMNI_LOGGER_INFO(__FILE__, __LINE__,
// __FUNCTION__, __VA_ARGS__)
