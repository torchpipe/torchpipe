#pragma once

#include <any>
#include <typeinfo>
#include <utility>

#include <tvm/ffi/object.h>
#include <tvm/ffi/any.h>
#include <tvm/ffi/string.h>



namespace omniback::ffi {

class AnyObj : public tvm::ffi::Object {
 public:
  std::any data;
  
  AnyObj()=default;
  template <typename T>
  explicit AnyObj(T&& data) : data(std::forward<T>(data)) {
    static_assert(
        !::tvm::ffi::TypeTraits<std::decay_t<T>>::convert_enabled &&
            !std::is_same_v<std::decay_t<T>, ::tvm::ffi::Any>);
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

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("omniback.Any", AnyObj, tvm::ffi::Object);
};

class Any : public tvm::ffi::ObjectRef {
 public:
  Any() = delete;


  template <
    typename T,
    typename Decayed = std::decay_t<T>,
    std::enable_if_t<
        !std::is_same_v<Decayed, ::tvm::ffi::Any> &&
        !std::is_same_v<Decayed, Any>,  // 排除自身（假设在 omniback::ffi 命名空间）
        int> = 0>
  explicit Any(T&& data) {
    data_ = ::tvm::ffi::make_object<AnyObj>(std::forward<T>(data));
  }

  std::any& get_data() {
    return static_cast<AnyObj*>(data_.get())->data;
  }

  const std::any& get_data() const {
    return static_cast<const AnyObj*>(data_.get())->data;
  }

  const std::type_info& type() const noexcept{
    return get()->type();
  }

  // Required: define object reference methods
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(Any, tvm::ffi::ObjectRef, AnyObj);
};

} // namespace omniback::ffi