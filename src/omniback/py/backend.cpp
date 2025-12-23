
#include "omniback/core/backend.hpp"
#include "tvm/ffi/reflection/registry.h"
#include "omniback/ffi/dict.h"
#include "tvm/ffi/container/variant.h"

namespace omniback::py {
using omniback ::Backend;
class BackendObj : public tvm::ffi::Object {
 public:
  std::shared_ptr<Backend> data;

  explicit BackendObj() {
    data = std::make_shared<Backend>();
  }

  explicit BackendObj(std::shared_ptr<Backend> ptr) : data(ptr) {
    TVM_FFI_ICHECK(data) << "null BackendObj is not allowed";
  }

  explicit BackendObj(std::unique_ptr<Backend>&& ptr) : data(std::move(ptr)) {
    TVM_FFI_ICHECK(data) << "null BackendObj is not allowed";
  }

  uint32_t max() const {
    return data->max();
  }
  uint32_t min() const {
    return data->min();
  }

//   operator std::shared_ptr<Backend>() const {
//     return data;
//   }

  static constexpr bool _type_mutable = true;
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL(
      "omniback.Backend",
      BackendObj,
      tvm::ffi::Object);
};

namespace tf = tvm::ffi;
namespace refl = tvm::ffi::reflection;

static std::shared_ptr<Backend> create(
    const std::string& class_name,
    tvm::ffi::Optional<tvm::ffi::String> aspect_name) {
    auto backend = std::shared_ptr<Backend>(std::move(create_backend(class_name)));
    if (aspect_name.has_value()) {
      register_backend(aspect_name.value(), backend);
    }
    return backend;
};

using omniback::ffi::DictObj ;
TVM_FFI_STATIC_INIT_BLOCK() {
  refl::GlobalDef().def("omniback.create", create);
  refl::ObjectDef<BackendObj>()
      .def(refl::init<>())
      .def(
          "init",
          [](BackendObj* self,
             const tvm::ffi::Map<tvm::ffi::String, tvm::ffi::String>& params,
             const tvm::ffi::Optional<tvm::ffi::Variant<
                 DictObj*,
                 tvm::ffi::Map<tvm::ffi::String, tvm::ffi::Any>>>& options) {
            if (!options.has_value()){
              self->data->init(
                  {params.begin(), params.end()},
                  nullptr);
              return self;
            }
            const auto& options_value =
                options.value();
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
      .def("min", &BackendObj::min);
};
} // namespace omniback::py

namespace tvm::ffi {
template <>
    inline constexpr bool use_default_type_traits_v<std::shared_ptr<omniback::Backend>> = false;
template <>
inline constexpr bool use_default_type_traits_v<std::unique_ptr<omniback::Backend>> = false;


template <>
    struct TypeTraits<std::shared_ptr<omniback::Backend>>
    : public TypeTraitsBase {
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
struct TypeTraits<std::unique_ptr<omniback::Backend>> : public TypeTraitsBase {
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
};
