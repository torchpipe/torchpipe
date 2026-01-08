
#include <tvm/ffi/reflection/registry.h>

namespace om::ffi {

bool use_cxx11_abi() {
#if defined(__GNUC__) && defined(_GLIBCXX_USE_CXX11_ABI)
#if _GLIBCXX_USE_CXX11_ABI
  return true;
#else
  return false;
#endif
#else
  return true;
#endif
}
// TVM_FFI_DLL_EXPORT_TYPED_FUNC(use_cxx11_abi, om::ffi::use_cxx11_abi);

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("omniback.use_cxx11_abi", om::ffi::use_cxx11_abi);
  refl::GlobalDef().def(
      "omniback.compiled_with_cxx11_abi", om::ffi::use_cxx11_abi);
}

} // namespace

