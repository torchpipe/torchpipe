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

#pragma once

#include <any>
#include <type_traits>
#include <utility>

namespace omniback {
template <typename T>
struct is_prohibited_type
    : std::disjunction<
          std::is_same<std::decay_t<T>, std::size_t>,
          std::is_same<std::decay_t<T>, const char*>,
          std::is_same<std::decay_t<T>, char*>,
          std::is_same<std::decay_t<T>, const unsigned char*>,
          std::is_same<std::decay_t<T>, unsigned char*>> {};

class any {
 public:
  // Default constructor
  any() = default;

  // Copy/move constructors
  any(const any&) = default;
  any(any&&) = default;

  // Templated constructor with prohibition check
  template <
      typename T,
      std::enable_if_t<
          !is_prohibited_type<T>::value &&
              !std::is_same_v<std::decay_t<T>, any>,
          int> = 0>
  any(T&& value) : impl_(std::forward<T>(value)) {}

  // Assignment operators
  any& operator=(const any&) = default;
  any& operator=(any&&) = default;

  template <
      typename T,
      std::enable_if_t<
          !is_prohibited_type<T>::value &&
              !std::is_same_v<std::decay_t<T>, any>,
          int> = 0>
  any& operator=(T&& value) {
    impl_ = std::forward<T>(value);
    return *this;
  }

  // Modifiers
  template <typename T, typename... Args>
  std::enable_if_t<!is_prohibited_type<T>::value, void> emplace(
      Args&&... args) {
    impl_.emplace<T>(std::forward<Args>(args)...);
  }

  void reset() noexcept {
    impl_.reset();
  }
  void swap(any& other) noexcept {
    impl_.swap(other.impl_);
  }

  // Observers
  bool has_value() const noexcept {
    return impl_.has_value();
  }
  const std::type_info& type() const noexcept {
    return impl_.has_value() ? impl_.type() : typeid(void);
  }

  // Friend declarations for any_cast
  template <typename T>
  friend T any_cast(const any&);
  template <typename T>
  friend T any_cast(any&);
  template <typename T>
  friend T any_cast(any&&);
  template <typename T>
  friend const T* any_cast(const any*) noexcept;
  template <typename T>
  friend T* any_cast(any*) noexcept;

 private:
  std::any impl_;
};

// any_cast implementations
template <typename T>
T any_cast(const any& operand) {
  return std::any_cast<T>(operand.impl_);
}

template <typename T>
T any_cast(any& operand) {
  return std::any_cast<T>(operand.impl_);
}

template <typename T>
T any_cast(any&& operand) {
  return std::any_cast<T>(std::move(operand.impl_));
}

template <typename T>
const T* any_cast(const any* operand) noexcept {
  return std::any_cast<T>(&operand->impl_);
}

template <typename T>
T* any_cast(any* operand) noexcept {
  return std::any_cast<T>(&operand->impl_);
}
} // namespace omniback