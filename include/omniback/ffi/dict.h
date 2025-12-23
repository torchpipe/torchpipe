#ifndef OMNIBACK_FFI_DICT_H__
#define OMNIBACK_FFI_DICT_H__



#include <unordered_map>
#include <memory>
#include <string>

// #include <tvm/ffi/error.h>


// #include <tvm/ffi/function.h>
// #include "omniback/ffi/dict.h"
// #include <tvm/ffi/function.h>
#include "omniback/core/any.hpp"
#include <tvm/ffi/reflection/registry.h>
// #include "tvm/ffi/extra/stl.h"
#include <tvm/ffi/type_traits.h>


namespace omniback::ffi {

namespace ffi = tvm::ffi;
namespace refl = tvm::ffi::reflection;

/*!
 * \brief 自定义字典对象，暴露给FFI
 */
class DictObj : public ffi::Object {
 public:
  std::shared_ptr<std::unordered_map<std::string, omniback::any>> data;

  static constexpr bool _type_mutable = true;
  DictObj()
      : data(std::make_shared<
             std::unordered_map<std::string, omniback::any>>()) {}
  explicit DictObj(
      std::shared_ptr<std::unordered_map<std::string, omniback::any>> in_data)
      : data(std::move(in_data)) {
    TVM_FFI_ICHECK(data) << "null DictObj is not allowed";
  }

  explicit DictObj(tvm::ffi::Map<tvm::ffi::String, tvm::ffi::Any> params) {
    data = std::make_shared<std::unordered_map<std::string, omniback::any>>(
        params.begin(), params.end());
  }

  /*!
   * \brief 获取可变map引用（懒初始化）
   * \return map的可变引用
   */
  std::unordered_map<std::string, omniback::any>& GetMutableMap() {
    if (!data) {
      data = std::make_shared<std::unordered_map<std::string, omniback::any>>();
    }
    return *data;
  }

  /*!
   * \brief 获取只读map引用
   * \return map的只读引用
   */
  const std::unordered_map<std::string, omniback::any>& GetMap() const {
    return *data;
  }
  std::shared_ptr<std::unordered_map<std::string, omniback::any>> get() const{
    return data;
  }
  operator std::shared_ptr<std::unordered_map<std::string, omniback::any>>() {
    return data;
  }

  // Required: declare type information
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("omniback.Dict", DictObj, ffi::Object);
};
} // namespace omniback::ffi

namespace tvm::ffi {
template <>
inline constexpr bool use_default_type_traits_v<
    std::shared_ptr<std::unordered_map<std::string, omniback::any>>> = false;

template <>
struct TypeTraits<
    std::shared_ptr<std::unordered_map<std::string, omniback::any>>>
    : public TypeTraitsBase {
 public:
  static constexpr bool storage_enabled = false;
  using Self = std::shared_ptr<std::unordered_map<std::string, omniback::any>>;
  using DictObj = omniback::ffi::DictObj;

  TVM_FFI_INLINE static void MoveToAny(Self&& src, TVMFFIAny* result) {
    if (!src) {
      tvm::ffi::TypeTraits<std::nullptr_t>::MoveToAny(nullptr, result);
    } else {
      auto data = tvm::ffi::make_object<DictObj>(std::move(src));
      tvm::ffi::TypeTraits<DictObj*>::MoveToAny(data.get(), result);
    }
  }

  TVM_FFI_INLINE static std::string TypeStr() {
    return "omniback::Dict";
  }
  TVM_FFI_INLINE static std::string TypeSchema() {
    return R"({"type":"omniback::Dict"})";
  }
};

}; // namespace tvm::ffi
#endif