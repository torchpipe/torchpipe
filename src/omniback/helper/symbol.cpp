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

#include <memory>
#include <sstream>

#include "omniback/helper/base_logging.hpp"

#include "omniback/helper/symbol.hpp"

namespace omniback {

#ifdef __GNUG__
#include <cxxabi.h>
#include <cstdlib>
#include <memory>
std::string local_demangle(const char* name) {
  int status = -1;
  std::unique_ptr<char, void (*)(void*)> res{
      abi::__cxa_demangle(name, NULL, NULL, &status), std::free};

  return (status == 0) ? res.get() : name;
}
#else

// does nothing if not g++
std::string local_demangle(const char* name) {
  return name;
}
#endif

void throw_wrong_type(const char* need_type, const char* input_type) {
  std::stringstream ss;
  ss << "get data of type " << local_demangle(input_type) << ", but we need "
     << local_demangle(need_type) << ".";
  SPDLOG_ERROR(ss.str());
  throw std::invalid_argument(ss.str());
}

void throw_not_exist(std::string key) {
  std::stringstream ss;
  ss << "key " << key << " not exist.";
  SPDLOG_ERROR(ss.str());
  throw std::invalid_argument(ss.str());
}

} // namespace omniback
