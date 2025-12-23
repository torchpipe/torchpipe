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

 
#ifndef OMNIBACK_ANY_H_
#define OMNIBACK_ANY_H_

#include <any>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <variant>
#include <iostream>

#include <tvm/ffi/base_details.h>
#include <tvm/ffi/c_api.h>
#include <tvm/ffi/object.h>
#include <tvm/ffi/type_traits.h>

#include "tvm/ffi/any.h"
#include "omniback/ffi/any.h"
// #include "omniback/types/basic.h"

namespace omniback::detail {

class Any {
 private:
  tvm::ffi::Any storage_; // todo

  friend struct tvm::ffi::details::AnyUnsafe;

  template <typename T>
  static constexpr bool is_tvm_convertible_v =
      tvm::ffi::TypeTraits<std::decay_t<T>>::convert_enabled ||
      std::is_same_v<std::decay_t<T>, ::tvm::ffi::Any>;

  template <typename T>
  void construct(T&& value) {
    if constexpr (is_tvm_convertible_v<T>) {
      storage_ = std::forward<T>(value);
    } else {
      storage_ = omniback::ffi::Any(std::forward<T>(value));
    }
  }

 public:

  operator tvm::ffi::AnyView() const { // NOLINT(google-explicit-constructor)
    return storage_;
  }

  operator tvm::ffi::Any() const { // NOLINT(google-explicit-constructor)
    return storage_;
  }

  TVMFFIAny MoveAnyToTVMFFIAny() {
    return tvm::ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(
        std::move(storage_));
  }

  public : Any() = default;


  template <typename T>
  std::optional<T> try_cast() const {
    if constexpr (
        ::tvm::ffi::TypeTraits<T>::convert_enabled ||
        std::is_same_v<T, ::tvm::ffi::Any>) {
      return storage_.try_cast<T>();
    } else {
      auto wrapper = storage_.try_cast<omniback::ffi::Any>();
      if (!wrapper) return std::nullopt;
      return wrapper.value()->try_cast<T>();
    }
  }

  template <typename T>
  T cast() const {
    if constexpr (
        ::tvm::ffi::TypeTraits<T>::convert_enabled ||
        std::is_same_v<T, ::tvm::ffi::Any>) {
      return storage_.cast<T>();
    } else {
      return storage_.cast<omniback::ffi::Any>()->cast<T>();
    }
  }

  template <
      typename T,
      std::enable_if_t<!std::is_same_v<std::decay_t<T>, Any>, bool> = true>
  Any(T&& value) {
    construct(std::forward<T>(value));
  }

  Any(const Any& other) = default;
  Any(Any&& other) = default;
  Any& operator=(const Any& other) = default;
  Any& operator=(Any&& other) = default;

  // bool has_value() const noexcept {
  //   return storage_ != nullptr;
  // }

  void reset() {
    storage_.reset();
  }

  std::string type_name() const {
    return storage_.GetTypeKey();
  }
};

template <typename T>
T any_cast(const Any& operand){
  return operand.cast<T>();
}

template <typename T>
Any make_any(T&& value) {
  return Any(std::forward<T>(value));
}

// template <typename T>
// bool is_type(const ::tvm::ffi::Any& operand) {
//   if constexpr (
//       ::tvm::ffi::TypeTraits<T>::convert_enabled ||
//       std::is_same_v<T, ::tvm::ffi::Any>) {
//     return ::tvm::ffi::details::AnyUnsafe::CheckAnyStrict<T>(operand);
//   }
//   else {
//     auto wrapper = operand.cast<omniback::ffi::Any>();
//     return typeid(T) == wrapper->type();
//   }
// }

} // namespace omniback::detail

namespace tvm::ffi 
{
template <>
inline constexpr bool use_default_type_traits_v<omniback::detail::Any> = false;

template <>
struct TypeTraits<omniback::detail::Any> : public TypeTraitsBase {
   public:
    using Self = omniback::detail::Any;

    TVM_FFI_INLINE static void CopyToAnyView(
        const Self& src,
        TVMFFIAny* result) {
      auto view = tvm::ffi::AnyView(src);
      *result = view.CopyToTVMFFIAny();
      std::cout << "CopyToAnyView Any" << std::endl;
    }
    TVM_FFI_INLINE static void MoveToAny(Self&& src, TVMFFIAny* result){
      *result = src.MoveAnyToTVMFFIAny();
      // std::cout << "MoveToAny Any" << std::endl;
      }

      TVM_FFI_INLINE static std::string TypeStr() {
        return "omniback::detail::any";
      }
      TVM_FFI_INLINE static std::string TypeSchema() {
        return R"({"type":"omniback::detail::any"})";
      }
  };

}; // namespace tvm::ffi

namespace omniback {
// using ffi::detail::is_type;
using detail::make_any;
using any = detail::Any;
using detail::any_cast;
} // namespace omniback
#endif // OMNIBACK_ANY_H_
 