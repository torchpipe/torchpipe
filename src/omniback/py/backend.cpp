
#include "omniback/core/backend.hpp"
#include "tvm/ffi/reflection/registry.h"
#include "omniback/ffi/dict.h"
#include "tvm/ffi/container/variant.h"
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/error.h>

#include "omniback/py/pybackend.hpp"
namespace omniback::py {
using omniback ::Backend;
class BackendObj : public tvm::ffi::Object {
 public:
  std::shared_ptr<Backend> data_owned;
  Backend* data;

  explicit BackendObj() {
    data_owned = std::make_shared<Backend>();
    data = data_owned.get();
  }

  explicit BackendObj(std::shared_ptr<Backend> ptr) : data_owned(ptr) {
    TVM_FFI_ICHECK(ptr) << "null BackendObj is not allowed";
    data = data_owned.get();
  }
  explicit BackendObj(Backend* ptr) : data(ptr) {
    TVM_FFI_ICHECK(ptr) << "null BackendObj is not allowed";
  }

  explicit BackendObj(std::unique_ptr<Backend>&& ptr)
      : data_owned(std::move(ptr)) {
    TVM_FFI_ICHECK(data) << "null BackendObj is not allowed";
    data = data_owned.get();
  }

  uint32_t max() const {
    return data->max();
  }
  uint32_t min() const {
    return data->min();
  }

  void inject_dependency(Backend* dep) {
    data->inject_dependency(dep);
  }

  static constexpr bool _type_mutable = true;
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL(
      "omniback.Backend",
      BackendObj,
      tvm::ffi::Object);
};

using omniback::ffi::DictObj;
namespace tf = tvm::ffi;
namespace refl = tvm::ffi::reflection;

namespace {
std::shared_ptr<Backend> pycreate(
    const std::string& class_name,
    tvm::ffi::Optional<tvm::ffi::String> aspect_name) {
  auto backend =
      std::shared_ptr<Backend>(std::move(create_backend(class_name)));
  if (aspect_name.has_value()) {
    register_backend(aspect_name.value(), backend);
  }
  TVM_FFI_ICHECK(backend);
  return backend;
};

using ParamsMap = tvm::ffi::Map<tvm::ffi::String, tvm::ffi::String>;
using OptionsDict = tvm::ffi::Optional<DictObj*>;
template <typename FType>
using TypedFunction = tvm::ffi::TypedFunction<FType>;

template <typename... Ts>
using Variant = tvm::ffi::Variant<Ts...>;
template <typename T>
using Optional = tvm::ffi::Optional<T>;
template <typename T>
using Array = tvm::ffi::Array<T>;

using Object = tvm::ffi::Object;

void pyregister(
    tvm::ffi::String name,
    SelfType py_obj,
    tvm::ffi::Optional<tvm::ffi::TypedFunction<void(
        SelfType,
        const tvm::ffi::Map<tvm::ffi::String, tvm::ffi::String>&,
        tvm::ffi::Optional<PyDictRef>)>> init_func,
    tvm::ffi::Optional<
        tvm::ffi::TypedFunction<void(SelfType, tvm::ffi::Array<PyDictRef>)>>
        forward_func,
    tvm::ffi::Optional<tvm::ffi::Variant<
        tvm::ffi::TypedFunction<uint32_t(SelfType)>,
        uint32_t>> max_func,
    tvm::ffi::Optional<tvm::ffi::Variant<
        tvm::ffi::TypedFunction<uint32_t(SelfType)>,
        uint32_t>> min_func) {
  if (const auto& v =
          py_obj
              .try_cast<tvm::ffi::TypedFunction<std::decay_t<SelfType>()>>()) {
    auto creator = v.value();
    std::function<Backend*()> f =
        [creator, init_func, forward_func, max_func, min_func]() {
          return object2backend(
                     creator(), init_func, forward_func, max_func, min_func)
              .release();
        };
    ClassRegistryInstance<Backend>().DoAddClass(name, f);
  } else {
    auto back =
        object2backend(py_obj, init_func, forward_func, max_func, min_func);
    register_backend(name, std::move(back));
  }
}

Backend* py_get_backend(const std::string& aspect_name_str) {
  return OMNI_INSTANCE_GET(Backend, aspect_name_str);
}

std::shared_ptr<Backend> pyinit(
    const tvm::ffi::String& class_config,
    const tvm::ffi::Map<tvm::ffi::String, tvm::ffi::String>& params,
    const tvm::ffi::Optional<tvm::ffi::Variant<
        DictObj*,
        tvm::ffi::Map<tvm::ffi::String, tvm::ffi::Any>>>& options,
    tvm::ffi::Optional<tvm::ffi::String> aspect_name) {
  auto backend = init_backend(
      class_config,
      {params.begin(), params.end()},
      [&]() -> std::shared_ptr<std::unordered_map<std::string, omniback::any>> {
        if (!options.has_value()) {
          return nullptr;
        }
        const auto& options_value = options.value();
        if (auto data = options_value.as<DictObj*>()) {
          TVM_FFI_ICHECK(data.value()) << "null DictObj* is not allowed";
          return data.value()->get();
        } else {
          auto map_data =
              options_value.as<tvm::ffi::Map<tvm::ffi::String, tvm::ffi::Any>>()
                  .value();
          return std::make_shared<
              std::unordered_map<std::string, omniback::any>>(
              map_data.begin(), map_data.end());
        }
      }(),
      aspect_name.has_value() ? aspect_name.value() : "");

  return std::move(backend);
};

void backend_forward_with_dep_function(
    BackendObj* self,
    const tvm::ffi::Variant<
        DictObj*,
        tvm::ffi::Array<DictObj*>,
        tvm::ffi::Map<tvm::ffi::String, tvm::ffi::Any>,
        tvm::ffi::Array<tvm::ffi::Map<tvm::ffi::String, tvm::ffi::Any>>>& ios,
    tvm::ffi::Optional<BackendObj*> dep) {
  if (auto data = ios.as<DictObj*>()) {
    const auto& item = data.value();

    TVM_FFI_ICHECK(item) << "null DictObj* is not allowed";
    DictObj::PyCallBackGuard guard(item);

    item->check_pycallback_legal();
    if (!dep.has_value())
      self->data->forward({item->get()});
    else {
      self->data->forward_with_dep({item->get()}, *(dep.value()->data));
    }
    item->try_invoke_and_clean_pycallback();
  } else if (auto data = ios.as<tvm::ffi::Array<DictObj*>>()) {
    TVM_FFI_ICHECK(data.value().size() > 0) << "empty input is not allowed";
    std::vector<omniback::dict> vec;
    DictObj::PyCallBackGuard guard;
    for (const auto& item : data.value()) {
      TVM_FFI_ICHECK(item) << "null DictObj* is not allowed";
      guard.add(item);
      item->check_pycallback_legal();
      vec.push_back(item->get());
    }
    if (!dep.has_value())
      self->data->forward(vec);
    else
      self->data->forward_with_dep(vec, *(dep.value()->data));

    for (const auto& item : data.value()) {
      item->try_invoke_and_clean_pycallback();
    }
  } else
    TVM_FFI_THROW(TypeError)
        << "invalid input type for Backend.__call__. Use om.Dict or List[om.Dict].";
};

void backend_forward(
    BackendObj* self,
    const tvm::ffi::Variant<
        DictObj*,
        tvm::ffi::Array<DictObj*>,
        tvm::ffi::Map<tvm::ffi::String, tvm::ffi::Any>,
        tvm::ffi::Array<tvm::ffi::Map<tvm::ffi::String, tvm::ffi::Any>>>& ios) {
  backend_forward_with_dep_function(self, ios, std::nullopt);
}

} // namespace

TVM_FFI_STATIC_INIT_BLOCK() {
  refl::GlobalDef().def("omniback.create", pycreate);
  refl::GlobalDef().def("omniback.init", pyinit);
  refl::GlobalDef().def("omniback.register", pyregister);
  refl::GlobalDef().def("omniback.get", py_get_backend);
  refl::GlobalDef().def("omniback.cleanup", []() { cleanup_backend(); });

  refl::ObjectDef<BackendObj>()
      .def(refl::init<>())
      .def(
          "_init",
          [](BackendObj* self,
             const tvm::ffi::Map<tvm::ffi::String, tvm::ffi::String>& params,
             const tvm::ffi::Optional<tvm::ffi::Variant<
                 DictObj*,
                 tvm::ffi::Map<tvm::ffi::String, tvm::ffi::Any>>>& options) {
            if (!options.has_value()) {
              self->data->init({params.begin(), params.end()}, nullptr);
              return self;
            }
            const auto& options_value = options.value();
            if (auto data = options_value.as<DictObj*>()) {
              TVM_FFI_ICHECK(data.value()) << "null DictObj* is not allowed";

              self->data->init(
                  {params.begin(), params.end()}, data.value()->get());
            } else {
              auto map_data =
                  options_value
                      .as<tvm::ffi::Map<tvm::ffi::String, tvm::ffi::Any>>()
                      .value();
              self->data->init(
                  {params.begin(), params.end()},
                  std::make_shared<
                      std::unordered_map<std::string, omniback::any>>(
                      map_data.begin(), map_data.end()));
            }
            return self;
          })
      .def("max", &BackendObj::max)
      .def("min", &BackendObj::min)
      .def("forward", &backend_forward)
      .def("forward_with_dep", &backend_forward_with_dep_function)
      .def("inject_dependency", [](BackendObj* self, BackendObj* dep) {
        self->data->inject_dependency(dep->data);
      });
};
} // namespace omniback::py

namespace tvm::ffi {
template <>
inline constexpr bool
    use_default_type_traits_v<std::shared_ptr<omniback::Backend>> = false;
template <>
inline constexpr bool
    use_default_type_traits_v<std::unique_ptr<omniback::Backend>> = false;

template <>
struct TypeTraits<std::shared_ptr<omniback::Backend>> : public TypeTraitsBase {
 public:
  static constexpr bool storage_enabled = false;
  using Self = std::shared_ptr<omniback::Backend>;
  using BackendObj = omniback::py::BackendObj;

  TVM_FFI_INLINE static void MoveToAny(Self&& src, TVMFFIAny* result) {
    if (!src) {
      tvm::ffi::TypeTraits<std::nullptr_t>::MoveToAny(nullptr, result);
    } else {
      auto data = tvm::ffi::make_object<BackendObj>(std::move(src));
      tvm::ffi::TypeTraits<BackendObj*>::MoveToAny(data.get(), result);
    }
  }

  TVM_FFI_INLINE static std::string TypeStr() {
    return "omniback::Backend";
  }
  TVM_FFI_INLINE static std::string TypeSchema() {
    return R"({"type":"omniback::Backend"})";
  }
};

template <>
struct TypeTraits<std::unique_ptr<omniback::Backend>>
    : public TypeTraits<std::shared_ptr<omniback::Backend>> {
 public:
  static constexpr bool storage_enabled = false;
  using Self = std::unique_ptr<omniback::Backend>;
  using BackendObj = omniback::py::BackendObj;

  TVM_FFI_INLINE static void MoveToAny(Self&& src, TVMFFIAny* result) {
    if (!src) {
      tvm::ffi::TypeTraits<std::nullptr_t>::MoveToAny(nullptr, result);
    } else {
      auto data = tvm::ffi::make_object<BackendObj>(std::move(src));
      tvm::ffi::TypeTraits<BackendObj*>::MoveToAny(data.get(), result);
    }
  }
};

template <>
struct TypeTraits<omniback::Backend*>
    : public TypeTraits<std::shared_ptr<omniback::Backend>> {
 public:
  static constexpr bool storage_enabled = false;
  using Self = omniback::Backend*;
  using BackendObj = omniback::py::BackendObj;

  TVM_FFI_INLINE static void MoveToAny(Self&& src, TVMFFIAny* result) {
    if (!src) {
      tvm::ffi::TypeTraits<std::nullptr_t>::MoveToAny(nullptr, result);
    } else {
      auto data = tvm::ffi::make_object<BackendObj>(src);
      tvm::ffi::TypeTraits<BackendObj*>::MoveToAny(data.get(), result);
    }
  }
};
}; // namespace tvm::ffi
