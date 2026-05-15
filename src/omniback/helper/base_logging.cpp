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

#include "spdlog/fmt/bundled/color.h"
#include "spdlog/fmt/fmt.h"
#include "spdlog/spdlog.h"

#include "omniback/helper/base_logging.hpp"
#include <mutex>

namespace om {

std::shared_ptr<spdlog::logger> default_logger() {
  return spdlog::default_logger();
}

spdlog::logger* default_logger_raw() {
  return spdlog::default_logger_raw();
}

std::string colored(const std::string& message) {
  return fmt::format(
      fmt::bg(fmt::terminal_color::cyan) |
          fmt::fg(fmt::terminal_color::black) |
          fmt::emphasis::bold,
      message
  );
}

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

} // namespace om