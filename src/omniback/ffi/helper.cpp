
#include <tvm/ffi/reflection/registry.h>

namespace omniback::ffi {

bool use_cxx11_abi() {
#if _GLIBCXX_USE_CXX11_ABI
    return true;
#else
    return false;
#endif
}
TVM_FFI_DLL_EXPORT_TYPED_FUNC(use_cxx11_abi, omniback::ffi::use_cxx11_abi);

} // namespace

