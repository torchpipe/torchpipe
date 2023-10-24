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
#include <vector>


// #include "MatView.hpp"
namespace ipipe {

template <class T, typename std::enable_if<(std::is_arithmetic<T>::value), int>::type = 0>
bool deserialize(std::vector<T>& in, const char* data, std::size_t len) {
  if (sizeof(T) * in.size() != len) {
    throw std::runtime_error("deserialize: sizeof(T) * in.size() != len");
  }
  const T* iter = (const T*)data;
  in = std::vector<T>(iter, iter + len);
}

template <class T, typename std::enable_if<(std::is_arithmetic<T>::value), int>::type = 0>
bool deserialize(T& in, const char* data, std::size_t len) {
  if (sizeof(T) != len) {
    throw std::runtime_error("deserialize: sizeof(T) != len");
  }
  in = *(T*)data;
}

template <class T, typename std::enable_if<(!std::is_arithmetic<T>::value), int>::type = 0>
bool deserialize(T& in, const char* data, std::size_t len) {
  return false;
}

template <class T, typename std::enable_if<(std::is_arithmetic<T>::value), int>::type = 0>
std::vector<char> serialize(const std::vector<T>& in, bool& success) noexcept {
  const char* start = (const char*)in.data();
  success = true;
  return std::vector<char>(start, start + in.size() * sizeof(T));
}

template <class T, typename std::enable_if<(std::is_arithmetic<T>::value), int>::type = 0>
std::vector<char> serialize(const T& in, bool& success) noexcept {
  return serialize<T>({in}, success);
}

template <class T, typename std::enable_if<(!std::is_arithmetic<T>::value), int>::type = 0>
std::vector<char> serialize(const T& in, bool& success) noexcept {
  success = false;
  //   throw std::runtime_error("unable to serialize " + std::string(typeid(in).name()));
  return {};
}


};  // namespace ipipe