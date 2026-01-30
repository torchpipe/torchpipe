// #include <unordered_map>
// #include <string>
// #include <functional>

#include "omniback/core/group.hpp"
#include "omniback/py/function_holder.hpp"
#include "omniback/ffi/cleanup.h"

#include <tvm/ffi/reflection/registry.h>
// #include <tvm/ffi/function.h>
// #include <tvm/ffi/string.h>
#include <tvm/ffi/extra/stl.h>


namespace om::ffi {

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("omniback.register_backend_group", om::register_backend_group);

  om::ffi::atexit_register(om::clear_groups);

}
// TVM_FFI_DLL_EXPORT_TYPED_FUNC(example, example)

}