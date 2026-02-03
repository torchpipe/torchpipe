#pragma once
#include "omniback/helper/box.hpp"
#include <tvm/ffi/object.h>
#include <tvm/ffi/type_traits.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ffi/extra/stl.h>

namespace om::ffi{
struct BoxObj : public tvm::ffi::Object {
  int id; // Class ID
  float score; // Confidence score
  float x1, y1, x2, y2; // Coordinates in XYXY format
  BoxObj() = default;
  BoxObj(Box box) {
    id = box.id;
    score = box.score;
    x1 = box.x1;
    y1 = box.y1;
    x2 = box.x2;
    y2 = box.y2;
  }
  
  Box get_box() {
    return Box{id, score, x1, y1, x2, y2};
  }

  // static constexpr bool _type_mutable = true;

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL(
      /*type_key=*/"omniback.Box",
      /*class=*/BoxObj,
      /*parent_class=*/tvm::ffi::Object);
};

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<BoxObj>()
      .def(refl::init<>())
      .def_ro("id", &BoxObj::id)
      .def_ro("score", &BoxObj::score)
      .def_ro("x1", &BoxObj::x1)
      .def_ro("y1", &BoxObj::y1)
      .def_ro("x2", &BoxObj::x2)
      .def_ro("y2", &BoxObj::y2);
}
}

namespace tvm::ffi {
template <>
inline constexpr bool use_default_type_traits_v<om::Box> = false;

using om::ffi::BoxObj;

template <>
struct TypeTraits<om::Box> : public TypeTraits<BoxObj*> {
 public:
  using Self = om::Box;
  
      // TVM_FFI_INLINE static void CopyToAnyView(
      //     const Self& src,
      //     TVMFFIAny* result) {
      //   auto view = tvm::BoxView(src);
      //   *result = view.CopyToTVMFFIAny();
      // }

  TVM_FFI_INLINE static void MoveToAny(Self&& src, TVMFFIAny* result) {
    auto data = tvm::ffi::make_object<BoxObj>(std::move(src));
    tvm::ffi::TypeTraits<BoxObj*>::MoveToAny(data.get(), result);
  }
  // TVM_FFI_INLINE static std::optional<Self> TryCastFromAnyView(
  //     const TVMFFIAny* src) {
  //   std::optional<om::ffi::BoxObj*> re =
  //       tvm::ffi::TypeTraits<BoxObj*>::TryCastFromAnyView(src);
  //   if (re.has_value()) {
  //     return re.value()->get_box();
  //   } else {
  //     return std::nullopt;
  //   }
  // }

  TVM_FFI_INLINE static std::string TypeStr() {
    return "om::Box";
  }
  TVM_FFI_INLINE static std::string TypeSchema() {
    return R"({"type":"om::Box"})";
  }
};

}; // namespace tvm::ffi