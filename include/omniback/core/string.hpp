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

#ifndef __OMNI_STRING_H__
#define __OMNI_STRING_H__
#include <string>
#include "omniback/helper/string.hpp"

namespace omniback {
// using string = std::string;
// using str_map = std::unordered_map<string, string>;
// using mapmap = std::unordered_map<string, str_map>;
using string = str::string;
} // namespace omniback
#endif