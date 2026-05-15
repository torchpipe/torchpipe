#ifndef OMNIBACK_FFI_DICT_H__
#define OMNIBACK_FFI_DICT_H__

#include <unordered_map>
#include <memory>
#include <string>

#include "omniback/ffi/any_wrapper.h"
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ffi/type_traits.h>
#include <tvm/ffi/function.h>

namespace om::ffi {
using dict = std::shared_ptr<std::unordered_map<std::string, om::ffi::Any>>;

namespace ffi = tvm::ffi;
namespace refl = tvm::ffi::reflection;

/*!
 * \brief Custom dictionary object exposed to FFI
 */
class DictObj : public ffi::Object {
 public:
  std::shared_ptr<std::unordered_map<std::string, om::any>> data;
  tvm::ffi::Function py_callback;

  struct PyCallBackGuard {
    PyCallBackGuard(DictObj* dict_obj) {
      add(dict_obj);
    }
    PyCallBackGuard() = default;

    void add(DictObj* dict_obj) {
      TVM_FFI_ICHECK(dict_obj);
      dict_objects_.push_back(dict_obj);
    }

    ~PyCallBackGuard() {
      for (const auto& dict_obj : dict_objects_) {
        dict_obj->clean_pycallback();
      }
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
             std::unordered_map<std::string, om::any>>()) {}
  explicit DictObj(
      std::shared_ptr<std::unordered_map<std::string, om::any>> in_data)
      : data(std::move(in_data)) {
    TVM_FFI_ICHECK(data) << "null DictObj is not allowed";
  }

  explicit DictObj(tvm::ffi::Map<tvm::ffi::String, tvm::ffi::Any> params) {
    data = std::make_shared<std::unordered_map<std::string, om::any>>(
        params.begin(), params.end());
  }

  /*!
   * \brief Get mutable map reference (lazy initialization)
   * \return Mutable reference to the map
   */
  std::unordered_map<std::string, om::any>& GetMutableMap() {
    if (!data) {
      data = std::make_shared<std::unordered_map<std::string, om::any>>();
    }
    return *data;
  }

  /*!
   * \brief Get read-only map reference
   * \return Read-only reference to the map
   */
  const std::unordered_map<std::string, om::any>& GetMap() const {
    return *data;
  }
  std::shared_ptr<std::unordered_map<std::string, om::any>> get() const {
    return data;
  }
  operator std::shared_ptr<std::unordered_map<std::string, om::any>>() {
    return data;
  }

  // Required: declare type information
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("omniback.Dict", DictObj, ffi::Object);
};

class DictRef : public tvm::ffi::ObjectRef {
 public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(
      DictRef,
      tvm::ffi::ObjectRef,
      DictObj);
};

} // namespace om::ffi

namespace tvm::ffi {

template <>
struct TypeTraits<
    std::shared_ptr<std::unordered_map<std::string, om::any>>>
    : public TypeTraits<om::ffi::DictObj*> {
 public:
  using Self = std::shared_ptr<std::unordered_map<std::string, om::any>>;
  using DictObj = om::ffi::DictObj;

  TVM_FFI_INLINE static void MoveToAny(Self&& src, TVMFFIAny* result) {
    if (!src) {
      tvm::ffi::TypeTraits<std::nullptr_t>::MoveToAny(nullptr, result);
    } else {
      auto data = tvm::ffi::make_object<DictObj>(std::move(src));
      tvm::ffi::TypeTraits<DictObj*>::MoveToAny(data.get(), result);
    }
  }

  TVM_FFI_INLINE static std::optional<Self> TryCastFromAnyView(const TVMFFIAny* src) {
    std::optional<om::ffi::DictObj*> re = tvm::ffi::TypeTraits<DictObj*>::TryCastFromAnyView(src);
    if (re.has_value()) {
      return re.value()->get();
    }
    return std::nullopt;
  }

  TVM_FFI_INLINE static std::string TypeStr() {
    return "om::Dict";
  }
  TVM_FFI_INLINE static std::string TypeSchema() {
    return R"({"type":"om::Dict"})";
  }
};

} // namespace tvm::ffi
#endif