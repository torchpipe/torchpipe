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
#include <unordered_map>
#include <memory>

#include <tvm/ffi/base_details.h>
#include <tvm/ffi/c_api.h>
#include <tvm/ffi/object.h>
#include <tvm/ffi/type_traits.h>
#include <tvm/ffi/error.h>

#include "tvm/ffi/any.h"
#include "omniback/ffi/any.h"
#include "omniback/ffi/type_traits.h"

// #include "omniback/types/basic.h"

namespace omniback::ffi {

using AnyStorage =
    std::variant<tvm::ffi::Any, StdAny, uint64_t>;

class Any {
 private:
  AnyStorage storage_;

  template <typename T>
  static constexpr bool is_tvm_type_v =
      tvm::ffi::TypeTraits<std::decay_t<T>>::convert_enabled ||
      std::is_same_v<std::decay_t<T>, ::tvm::ffi::Any>;

  template <typename T>
  static constexpr bool is_om_type_v =
      omniback::ffi::OmTypeTraits<std::decay_t<T>>::convert_enabled;

  template <typename T>
  void construct(T&& value) {
    using TDecay = std::decay_t<T>;

    static_assert(
        !(std::is_integral_v<TDecay> && sizeof(TDecay) > sizeof(int64_t)));
    if constexpr (
        std::is_unsigned_v<TDecay> && sizeof(TDecay) == sizeof(uint64_t)) {
      if (value > static_cast<int64_t>(INT64_MAX)) {
        storage_ = static_cast<uint64_t>(value);
      } else {
        storage_ = tvm::ffi::Any(std::forward<T>(value));
      }
    } else if constexpr (is_om_type_v<T>) {
      storage_ = StdAny(std::forward<T>(value));
    } else if constexpr (is_tvm_type_v<T>) {
      storage_ = tvm::ffi::Any(std::forward<T>(value));
    } else {
      storage_ = StdAny(std::forward<T>(value));
    }
  }

 public:
  Any() = default;

  template <
      typename T,
      std::enable_if_t<!std::is_same_v<std::decay_t<T>, Any>, int> = 0>
  Any(T&& value) {
    construct(std::forward<T>(value));
  }

  Any(const Any& other) = default;
  Any(Any&& other) = default;
  Any& operator=(const Any& other) = default;
  Any& operator=(Any&& other) = default;


  template <typename T>
  std::optional<T> try_cast() const {
    using TDecay = std::decay_t<T>;

    std::cout << "Current index: " << storage_.index() << std::endl;
    if constexpr (is_tvm_type_v<TDecay>)
    {
    if (const auto* v = std::get_if<tvm::ffi::Any>(&storage_)) {
        return v->try_cast<TDecay>();
      } else if (const auto* v = std::get_if<StdAny>(&storage_)) {
        return (*v)->try_cast<TDecay>();
      } else {
        if constexpr (std::is_integral_v<TDecay>)
          return static_cast<TDecay>(std::get<uint64_t>(storage_));
        else {
          return std::nullopt;
        }
      }
    }
  }

  template <typename T>
  T cast() const {
    using TDecay = std::decay_t<T>;

    std::cout << "Current index: " << storage_.index() << std::endl;

    if (const auto* v = std::get_if<tvm::ffi::Any>(&storage_)) {
      return v->cast<TDecay>();
    } else if (const auto* v = std::get_if<StdAny>(&storage_)) {
      return (*v)->cast<TDecay>();
    } else {
      if constexpr (std::is_integral_v<TDecay>)
        return static_cast<TDecay>(std::get<uint64_t>(storage_));
      else{
        TVM_FFI_THROW("Cannot cast uint64_t storage to non-integral type");
      }
    }
  }

  // operator tvm::ffi::Any() const {
  //   if (auto* v = std::get_if<tvm::ffi::Any>(&storage_)) {
  //     return *v;
  //   }
  //   return make_tvm_any_from_storage();
  // }

  TVMFFIAny MoveAnyToTVMFFIAny() && {
    if (auto* v = std::get_if<tvm::ffi::Any>(&storage_)) {
      return tvm::ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(std::move(*v));
    }
    tvm::ffi::Any tmp = make_tvm_any_from_storage();
    return tvm::ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(std::move(tmp));
  }

  // --- Utils ---

  void reset() {
    storage_ = AnyStorage{};
  }

  std::string type_name() const {
    if (std::holds_alternative<tvm::ffi::Any>(storage_)) {
      return std::get<tvm::ffi::Any>(storage_).GetTypeKey();
    } else if (std::holds_alternative<StdAny>(storage_)) {
      return "StdAny";
    } else if (std::holds_alternative<uint64_t>(storage_)) {
      return "uint64_t";
    } 
    return "unknown";
  }

 private:
  tvm::ffi::Any make_tvm_any_from_storage() {
    if (std::holds_alternative<StdAny>(storage_)) {
      const auto& std_any = std::get<StdAny>(storage_);
      if (std_any->to_tvm_ffi_any_func){
        return std_any->to_tvm_ffi_any_func();
      } else
        return tvm::ffi::Any(std_any); 
    } else if (std::holds_alternative<uint64_t>(storage_)) {
      return tvm::ffi::Any(int64_t(std::get<uint64_t>(storage_))); //todo
    } else {
      throw std::runtime_error(
          "Unsupported non-TVM type in make_tvm_any_from_storage");
    }
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
//     auto wrapper = operand.cast<omniback::ffi::StdAny>();
//     return typeid(T) == wrapper->type();
//   }
// }

} // namespace omniback::ffi

namespace tvm::ffi 
{
template <>
inline constexpr bool use_default_type_traits_v<omniback::ffi::Any> = false;

template <>
struct TypeTraits<omniback::ffi::Any> : public TypeTraitsBase {
   public:
    using Self = omniback::ffi::Any;

    // TVM_FFI_INLINE static void CopyToAnyView(
    //     const Self& src,
    //     TVMFFIAny* result) {
    //   auto view = tvm::ffi::AnyView(src);
    //   *result = view.CopyToTVMFFIAny();
    // }

    TVM_FFI_INLINE static void MoveToAny(Self&& src, TVMFFIAny* result){
      *result = std::move(src).MoveAnyToTVMFFIAny();
      // std::cout << "MoveToAny Any" << std::endl;
      }

      TVM_FFI_INLINE static std::string TypeStr() {
        return "omniback::ffi::Any";
      }
      TVM_FFI_INLINE static std::string TypeSchema() {
        return R"({"type":"omniback::ffi::Any"})";
      }
  };

}; // namespace tvm::ffi

namespace omniback {
// using ffi::detail::is_type;
using ffi::make_any;
using any = ffi::Any;
using ffi::any_cast;
} // namespace omniback
#endif // OMNIBACK_ANY_H_
 