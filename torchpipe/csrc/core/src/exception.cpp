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

#include "exception.hpp"
#include "ipipe_common.hpp"
namespace ipipe {
std::exception_ptr add_exception(std::exception_ptr curr_except, const std::string& msg) {
  if (!curr_except) return nullptr;

  try {
    std::rethrow_exception(curr_except);
  } catch (...) {
    try {
      std::throw_with_nested(std::runtime_error(msg));
    } catch (...) {
      return std::current_exception();
    }
  }
}

std::exception_ptr add_exception(const std::string& msg) {
  try {
    std::throw_with_nested(std::runtime_error(msg));
  } catch (...) {
    return std::current_exception();
  }
  return nullptr;
}

std::exception_ptr insert_exception(const std::string& original_msg, const char* msg) {
  // std::throw_with_nested(std::runtime_error(msg));
  return std::make_exception_ptr(std::runtime_error(original_msg + "\nerror: " + msg));
}

std::exception_ptr insert_exception(const char* original_msg, const std::string& msg) {
  // std::throw_with_nested(std::runtime_error(msg));
  return std::make_exception_ptr(std::runtime_error(std::string(original_msg) + "\nerror: " + msg));
}

std::exception_ptr insert_exception(std::exception_ptr curr_except, const std::string& msg) {
  if (!curr_except) return nullptr;

  try {
    std::rethrow_exception(curr_except);
  } catch (const std::exception& e) {
    return insert_exception(e.what(), msg);
  } catch (...) {
    return insert_exception("an exception not drived from std::exception.", msg);
  }
  return nullptr;
}

}  // namespace ipipe