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

#include "reflect.h"
#include "spdlog/spdlog.h"
#include <stdexcept>
// to load the libipipe.so.
namespace ipipe {
namespace reflect {
bool ipipe_load() { return true; }

void printlog_and_throw(std::string name) {
  SPDLOG_ERROR(name);
  throw std::runtime_error(name);
}

}  // namespace reflect
}  // namespace ipipe
