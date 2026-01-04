#pragma once

#include <any>
#include <typeinfo>
#include <utility>
#include <functional>

#include <tvm/ffi/object.h>
#include <tvm/ffi/any.h>
#include <tvm/ffi/string.h>

#include "omniback/ffi/type_traits.h"


namespace omniback::ffi {

class StdAnyObj : public tvm::ffi::Object {
 public:
  std::any data;
  std::function<tvm::ffi::Any()> to_tvm_ffi_any_func;

  StdAnyObj()=default;

  template <
      typename T,
      std::enable_if_t<
          !::tvm::ffi::TypeTraits<std::decay_t<T>>::convert_enabled &&
              !std::is_same_v<std::decay_t<T>, ::tvm::ffi::Any>,
          int> = 0>
  explicit StdAnyObj(T&& data) : data(std::forward<T>(data)) {
  }

  template <
      typename T,
      std::enable_if_t<
          ::tvm::ffi::TypeTraits<std::decay_t<T>>::convert_enabled ||
              std::is_same_v<std::decay_t<T>, ::tvm::ffi::Any>,
          int> = 0>
  explicit StdAnyObj(T&& input_data)
      : to_tvm_ffi_any_func(
            [input_data]() { return tvm::ffi::Any(input_data); }),
        data(input_data) {
        }

  const std::type_info& type() const noexcept{
    return data.type();
  }

  template <typename T>
  std::optional<T> try_cast() const{
    if (typeid(T) == data.type()) {
      return std::any_cast<T>(data);
      }
    else {
      return std::nullopt;
    }
  }

  template <typename T>
  T cast() const{
    return std::any_cast<T>(data);
  }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("omniback.StdAny", StdAnyObj, tvm::ffi::Object);
};

class StdAny : public tvm::ffi::ObjectRef {
 public:
  StdAny() = delete;


  template <
    typename T,
    typename Decayed = std::decay_t<T>,
    std::enable_if_t<
        !std::is_same_v<Decayed, ::tvm::ffi::Any> &&
        !std::is_same_v<Decayed, StdAny>,  // 排除自身（假设在 omniback::ffi 命名空间）
        int> = 0>
  explicit StdAny(T&& data) {
    data_ = ::tvm::ffi::make_object<StdAnyObj>(std::forward<T>(data));
  }

  std::any& get_data() {
    return static_cast<StdAnyObj*>(data_.get())->data;
  }

  const std::any& get_data() const {
    return static_cast<const StdAnyObj*>(data_.get())->data;
  }

  const std::type_info& type() const noexcept{
    return get()->type();
  }

  // Required: define object reference methods
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(StdAny, tvm::ffi::ObjectRef, StdAnyObj);
};

} // namespace omniback::ffi

// namespace tvm::ffi {
// template <>
// inline constexpr bool use_default_type_traits_v<omniback::ffi::StdAny> = false;

// template <>
// struct TypeTraits<omniback::ffi::StdAny>
//     : public TypeTraits<tvm::ffi::ObjectRef> {
//  public:
//   using Self = omniback::ffi::StdAny;

//   // TVM_FFI_INLINE static void CopyToAnyView(
//   //     const Self& src,
//   //     TVMFFIAny* result) {
//   //   auto view = tvm::ffi::AnyView(src);
//   //   *result = view.CopyToTVMFFIAny();
//   // }

//   TVM_FFI_INLINE static void MoveToAny(Self&& src, TVMFFIAny* result) {
//     if (src->to_tvm_ffi_any_func){
//       auto tmp = src->to_tvm_ffi_any_func();
//       *result= tvm::ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(std::move(tmp));
//     }else{
//       *result = std::move(src)->MoveAnyToTVMFFIAny();
//     }
//     std::cout << "MoveToAny Any" << std::endl;
//   }

//   // TVM_FFI_INLINE static std::string TypeStr() {
//   //   return "omniback::ffi::Any";
//   // }
//   // TVM_FFI_INLINE static std::string TypeSchema() {
//   //   return R"({"type":"omniback::ffi::Any"})";
//   // }
// };

// }; // namespace tvm::ffi