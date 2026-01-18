#pragma once
#include <tvm/ffi/type_traits.h>
#include <tvm/ffi/container/tensor.h>

// #include <ATen/DLConvertor.h>
#include <ATen/Functions.h>
// #include <torch/extension.h>
#include <ATen/Tensor.h>
#include <torch/version.h>

#include "omniback/ffi/type_traits.h"



#include <ATen/ATen.h>
#include <ATen/dlpack.h>

#if TORCH_VERSION_MAJOR == 1 && TORCH_VERSION_MINOR <= 11
// for torch<=1.11. see https://github.com/pytorch/pytorch/issues/82823
namespace at {
 DLManagedTensor* toDLPack(const Tensor& src);
 Tensor fromDLPack(DLManagedTensor* src);
}
#else
#include <ATen/DLConvertor.h>
#endif

namespace om::ffi {

template <>
struct OmTypeTraits<at::Tensor>
    : public OmTypeTraitsBase {};
} // namespace om::ffi

namespace tvm::ffi {
template <>
struct TypeTraits<at::Tensor> : public TypeTraitsBase {
 public:
  // static constexpr bool storage_enabled = false;
  using Self = at::Tensor;

  TVM_FFI_INLINE static void MoveToAny(Self&& src, TVMFFIAny* result) {
#if defined(DLPACK_MAJOR_VERSION) && \
    (DLPACK_MAJOR_VERSION * 100 + DLPACK_MINOR_VERSION * 10 >= 130)
    DLManagedTensorVersioned* mid = ::at::toDLPackVersioned(src);
    tvm::ffi::Tensor te = tvm::ffi::Tensor::FromDLPackVersioned(mid);
#else
    DLManagedTensor* mid = ::at::toDLPack(src);
    tvm::ffi::Tensor te = tvm::ffi::Tensor::FromDLPack(mid);
#endif
    tvm::ffi::TypeTraits<tvm::ffi::Tensor>::MoveToAny(std::move(te), result);
  }

  TVM_FFI_INLINE static std::optional<Self> TryCastFromAnyView(
      const TVMFFIAny* src) {
    std::optional<tvm::ffi::Tensor> re =
        tvm::ffi::TypeTraits<tvm::ffi::Tensor>::TryCastFromAnyView(src);
#if defined(DLPACK_MAJOR_VERSION) && \
    (DLPACK_MAJOR_VERSION * 100 + DLPACK_MINOR_VERSION * 10 >= 130)
    if (re.has_value()) {
    return at::fromDLPackVersioned(re.value().ToDLPackVersioned());
  }
#else
    if (re.has_value()) {
      return at::fromDLPack(re.value().ToDLPack());
    }
#endif
    else {
      return std::nullopt;
    }
  }

  TVM_FFI_INLINE static std::string TypeStr() {
    return "at::Tensor";
  }
  TVM_FFI_INLINE static std::string TypeSchema() {
    return R"({"type":"at::Tensor"})";
  }
};

}; // namespace tvm::ffi