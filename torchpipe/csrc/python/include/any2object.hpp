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
#include "pybind11/pybind11.h"
#include "any.hpp"
#include "ipipe_common.hpp"

namespace ipipe {
inline pybind11::object IPIPE_EXPORT any2object_impl(const any& data);

// stop type conversion
template <typename T>
pybind11::object any2object(const T& data) {
  static_assert(std::is_same<T, any>::value, "The argument must be of type 'any'");
  return any2object_impl(data);
}
}  // namespace ipipe