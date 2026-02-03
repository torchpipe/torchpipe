#pragma once

#include <tvm/ffi/extra/stl.h>
#include "omniback/ffi/type_traits.h"
#include <utility>
// #include "omniback/ffi/any_wrapper.h"

namespace om::ffi {

template <typename F, typename S>
struct OmTypeTraits<std::pair<F, S>> : public OmTypeTraitsBase {};
}

namespace tvm {
namespace ffi {

template <typename T1, typename T2>
struct TypeTraits<std::pair<T1, T2>>
    : public TypeTraits<details::ListTemplate> {
 private:
  using Self = std::pair<T1, T2>;

  TVM_FFI_INLINE static bool CheckAnyFast(const TVMFFIAny* src) {
    if (src->type_index != TypeIndex::kTVMFFIArray)
      return false;
    const ArrayObj& arr = *reinterpret_cast<const ArrayObj*>(src->v_obj);
    return arr.size_ == 2;
  }

 public:
  static constexpr int32_t field_static_type_index = TypeIndex::kTVMFFIArray;

  TVM_FFI_INLINE static void CopyToAnyView(const Self& src, TVMFFIAny* result) {
    auto array = ArrayObj::Empty(2);
    auto dst = array->MutableBegin();
    // 异常安全：逐元素构造，失败时已构造元素会被 ArrayObj 析构
    ::new (dst) Any(src.first);
    array->size_++;
    ::new (dst + 1) Any(src.second);
    array->size_++;
    MoveToAnyImpl(std::move(array), result);
  }

  TVM_FFI_INLINE static void MoveToAny(Self&& src, TVMFFIAny* result) {
    auto array = ArrayObj::Empty(2);
    auto dst = array->MutableBegin();
    ::new (dst) Any(std::move(src.first));
    array->size_++;
    ::new (dst + 1) Any(std::move(src.second));
    array->size_++;
    MoveToAnyImpl(std::move(array), result);
  }

  TVM_FFI_INLINE static std::optional<Self> TryCastFromAnyView(
      const TVMFFIAny* src) {
    if (!CheckAnyFast(src))
      return std::nullopt;
    try {
      auto array = CopyFromAnyImpl<ArrayObj>(src);
      auto begin = array->MutableBegin();
      // 严格按顺序转换：first -> T1, second -> T2
      T1 first = ConstructFromAny<T1>(begin[0]);
      T2 second = ConstructFromAny<T2>(begin[1]);
      return Self{std::move(first), std::move(second)};
    } catch (const details::STLTypeMismatch&) {
      return std::nullopt;
    }
  }

  TVM_FFI_INLINE static std::string TypeStr() {
    return "std::pair<" + details::Type2Str<T1>::v() + ", " +
        details::Type2Str<T2>::v() + ">";
  }

  TVM_FFI_INLINE static std::string TypeSchema() {
    return R"({"type":"std::pair","args":[)" + details::TypeSchema<T1>::v() +
        "," + details::TypeSchema<T2>::v() + "]}";
  }
};

} // namespace ffi
} // namespace tvm