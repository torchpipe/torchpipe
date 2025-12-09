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

// #include "base_logging.hpp"

// #define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_DEBUG
#include "spdlog/fmt/bundled/color.h"
#include "spdlog/fmt/fmt.h"
#include "spdlog/spdlog.h"

#include "omniback/helper/base_logging.hpp"

namespace omniback {
std::shared_ptr<spdlog::logger> default_logger() {
  return spdlog::default_logger();
}
spdlog::logger* default_logger_raw() {
  return spdlog::default_logger_raw();
}

// Function to colorize text using spdlog and fmt
std::string colored(const std::string& message) {
  return fmt::format(
      fmt::bg(fmt::terminal_color::cyan) | // Set background color to cyan
          fmt::fg(fmt::terminal_color::black) | // Set foreground color to black
          fmt::emphasis::bold, // Set text to bold
      message // The message to format
  );
}
} // namespace omniback