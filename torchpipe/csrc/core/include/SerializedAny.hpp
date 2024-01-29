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
#include <vector>
#include "any.hpp"
namespace ipipe {
uint32_t get_unique_tag();
class SerializeAny {
 public:
  SerializeAny() = default;
  virtual bool serialize(const any& data, std::vector<char>& out) { return false; }
  virtual bool deserialize(any& out, const std::vector<char>& data) { return false; }
  virtual ~SerializeAny() = default;
};
};  // namespace ipipe