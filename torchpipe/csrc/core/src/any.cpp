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

#include "ipipe_utils.hpp"
#include <string>
#include "any.hpp"
#include <stdexcept>
namespace nonstd {
namespace any_lite {
std::string get_type_name(const std::type_info& info) {
  auto src = ipipe::local_demangle(info.name());
  auto iter = src.find(',');
  if (iter != std::string::npos) {
    src = src.substr(0, iter) + '>';
  }
  return src;
}

const char* bad_any_cast::what() const noexcept {
  // if (msg_.empty())
  {
    auto src = ipipe::local_demangle(src_.name());
    auto dst = ipipe::local_demangle(dst_.name());
    auto iter = src.find(',');
    if (iter != std::string::npos) {
      src = src.substr(0, iter) + '>';
    }
    iter = dst.find(',');
    if (iter != std::string::npos) {
      dst = dst.substr(0, iter);
    }

    msg_ = "any_cast<" + dst + ">(" + src + " input)";
  }

  return msg_.c_str();
}
}  // namespace any_lite
}  // namespace nonstd