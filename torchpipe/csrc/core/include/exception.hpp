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

#pragma once
#include <stdexcept>
#include <string>

namespace ipipe {
std::exception_ptr add_exception(std::exception_ptr curr_except, const std::string& msg);
std::exception_ptr add_exception(const std::string& msg);

std::exception_ptr insert_exception(const std::string& original_msg, const char* msg);
std::exception_ptr insert_exception(const char* original_msg, const std::string& msg);
std::exception_ptr insert_exception(std::exception_ptr curr_except, const std::string& msg);

#define TRACE_EXCEPTION(input)                                                               \
  {                                                                                          \
    try {                                                                                    \
      input;                                                                                 \
    } catch (const std::exception& e) {                                                      \
      const auto& trace_exception_msg =                                                      \
          ('[' + std::string(__FILE__) + ":" + std::to_string(__LINE__) + "]: failed:\n `" + \
           std::string(#input) + "`.");                                                      \
      throw std::runtime_error(std::string(e.what()) + "\nerror: " + trace_exception_msg);   \
    } catch (...) {                                                                          \
      const auto& trace_exception_msg =                                                      \
          ('[' + std::string(__FILE__) + ":" + std::to_string(__LINE__) + "]: failed:\n `" + \
           std::string(#input) + "`.");                                                      \
      throw std::runtime_error(std::string("an exception not drived from std::exception.") + \
                               "\nerror: " + trace_exception_msg);                           \
    }                                                                                        \
  }

#define RETURN_EXCEPTION_TRACE(input)                                                      \
  ([&]() {                                                                                   \
    try {                                                                                    \
      return input;                                                                          \
    } catch (const std::exception& e) {                                                      \
      const auto& trace_exception_msg =                                                      \
          ('[' + std::string(__FILE__) + ":" + std::to_string(__LINE__) + "]: failed:\n `" + \
           std::string(#input) + "`.");                                                      \
      throw std::runtime_error(std::string(e.what()) + "\nerror: " + trace_exception_msg);   \
    } catch (...) {                                                                          \
      const auto& trace_exception_msg =                                                      \
          ('[' + std::string(__FILE__) + ":" + std::to_string(__LINE__) + "]: failed:\n `" + \
           std::string(#input) + "`.");                                                      \
      throw std::runtime_error(std::string("an exception not drived from std::exception.") + \
                               "\nerror: " + trace_exception_msg);                           \
    }                                                                                        \
  }())

}  // namespace ipipe