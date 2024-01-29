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

#include <string>
#include <typeinfo>

// from
// https://stackoverflow.com/questions/281818/unmangling-the-result-of-stdtype-infoname
namespace ipipe {
std::string local_demangle(const char* name);

template <class T>
std::string type(const T& t) {
  return local_demangle(typeid(t).name());
}

}  // namespace ipipe
