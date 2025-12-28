#ifndef OMNIBACK_FFI_DICT_H__
#define OMNIBACK_FFI_DICT_H__



#include <unordered_map>
#include <memory>
#include <string>

// #include <tvm/ffi/error.h>


// #include <tvm/ffi/function.h>
// #include "omniback/ffi/dict.h"
// #include <tvm/ffi/function.h>
#include "omniback/ffi/any_wrapper.h"
#include <tvm/ffi/reflection/registry.h>
// #include "tvm/ffi/extra/stl.h"
#include <tvm/ffi/type_traits.h>
#include <tvm/ffi/function.h>

namespace omniback::ffi {
using dict = std::shared_ptr<std::unordered_map<std::string, omniback::ffi::Any>>;

namespace ffi = tvm::ffi;
namespace refl = tvm::ffi::reflection;

/*!
 * \brief 自定义字典对象，暴露给FFI
 */
class DictObj : public ffi::Object {
 public:
  std::shared_ptr<std::unordered_map<std::string, omniback::any>> data;
  tvm::ffi::Function py_callback;

  struct PyCallBackGuard{
    PyCallBackGuard(DictObj* dict_obj){
      add(dict_obj);
    }
    PyCallBackGuard() =default;

    void add(DictObj* dict_obj) {
      TVM_FFI_ICHECK(dict_obj);
      dict_objects_.push_back(dict_obj);
    }

    ~PyCallBackGuard(){
      for (const auto& dict_obj : dict_objects_)
        dict_obj->clean_pycallback();
      } 
    std::vector<DictObj*> dict_objects_;
  };

  void try_invoke_and_clean_pycallback() {
    if (py_callback.defined()) {
      TVM_FFI_ICHECK(data->find("event") == data->end())
          << "The 'event' key already exists in the dict; callback cannot be invoked. ";
      py_callback();
      py_callback = tvm::ffi::Function();
      TVM_FFI_ICHECK(!py_callback.defined()) << "callback should be cleared";
    }
  }

  void clean_pycallback() {

    py_callback = tvm::ffi::Function();
  }

  void check_pycallback_legal() {
    if (py_callback.defined()) {
      TVM_FFI_ICHECK(data->find("event") == data->end())
          << "If you are using asynchronous mode (i.e., the input dict contains an 'event' key), "
             "please use omniback.Dict instead of dict.";
    }
  }

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

class DictRef : public tvm::ffi::ObjectRef {
 public:
  // DictRef(const std::shared_ptr<std::unordered_map<std::string, omniback::any>>&data){
  //   data_ = tvm::ffi::make_object<DictObj>(data);
  // }
  // operator std::shared_ptr<std::unordered_map<std::string, omniback::any>>(){
  //   return data_->get();
  // }

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(
      DictRef,
      tvm::ffi::ObjectRef,
      DictObj);
};

} // namespace omniback::ffi

namespace tvm::ffi {
// template <>
// inline constexpr bool use_default_type_traits_v<
//     std::shared_ptr<std::unordered_map<std::string, omniback::any>>> = false;

    // ObjectRefTypeTraitsBase
    // TypeTraitsBase
template <>
struct TypeTraits<
    std::shared_ptr<std::unordered_map<std::string, omniback::any>>>
    : public TypeTraits<omniback::ffi::DictObj*> {
 public:
  // static constexpr bool storage_enabled = false;
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

  TVM_FFI_INLINE static std::optional<Self> TryCastFromAnyView(const TVMFFIAny* src) {
    std::optional<omniback::ffi::DictObj*> re = tvm::ffi::TypeTraits<DictObj*>::TryCastFromAnyView(src);
    if (re.has_value()){
      return re.value()->get();
    }
    else{
        return std::nullopt;
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