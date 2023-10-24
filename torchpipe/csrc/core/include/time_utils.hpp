// Copyright 2021-2023 NetEase.
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

#include <chrono>
#include <iostream>
#include <memory>
#include <type_traits>
#include <unordered_map>
#include "any.hpp"

namespace ipipe {
using namespace nonstd; // for any, any_cast

std::chrono::steady_clock::time_point inline now() {
  return std::chrono::steady_clock::now();
}

float inline time_passed(decltype(now()) time_old) {
  std::chrono::duration<float, std::milli> fp_ms = now() - time_old;
  return fp_ms.count();
}

float inline time_passed() {
  static std::chrono::steady_clock::time_point _basetime = now();

  std::chrono::duration<float, std::milli> fp_ms = now() - _basetime;
  return fp_ms.count();
}

float inline time_passed(any time_old) {
  auto time_old_value = any_cast<decltype(now())>(time_old);
  std::chrono::duration<float, std::milli> fp_ms = now() - time_old_value;
  return fp_ms.count();
}

/// 辅助的RAII型计时工具
class time_guard {
 public:
  time_guard(
      std::string message = "",
      bool print = true,
      uint32_t time_out = UINT32_MAX /*ms*/)
      : starttime_(now()), message_(message), time_out_(time_out) {
    bPrint = print;

    bStopped = false;
    index = 0;
  }
  time_guard& operator=(const time_guard&) = delete;
  time_guard(const time_guard&) = delete;

  ~time_guard() {
    stop();
  }
  time_guard& add(const std::string& info) {
    message_ += " || " + info;
    return *this;
  }
  void silence() {
    bStopped = true;
  }
  float time() {
    return time_passed(starttime_);
  }
  void stop()  /*超时强制宕机*/ {
    if (bStopped)
      return;
    bStopped = true;

    auto time_used = time_passed(starttime_);
    std::string str_time = std::to_string(int(10 * time_used) / 10.0);
    str_time.erase(str_time.find_last_not_of("0") + 1);
    if (!message_.empty())
      message_ += " || time used: " + str_time;
    else
      message_ += str_time;
    if (bPrint)
      std::cout << message_ << std::endl;
    // if (time_used > time_out_)
    //   throw std::runtime_error("timeout: " + message_); // 强制宕机
  }

  void reset(std::string message = "") {
    stop();
    start(message);
  }

  void start(std::string message = "") {
    message_ = message;
    bStopped = false;
  }

  const std::string& release()  {
    bPrint = false;
    time_out_ = UINT32_MAX;
    stop();
    return message_;
  }

 private:
  std::chrono::steady_clock::time_point starttime_;
  std::string message_;
  bool bStopped;
  int index;
  bool bPrint;
  uint32_t time_out_;
};

} // namespace ipipe