#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/function.h>
#include <omniback/ffi/queue.h>
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

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<FFIQueueObj>()
      .def(refl::init<>())
      .def("size", &FFIQueueObj::size)
      .def("empty", &FFIQueueObj::empty)
      .def("clear", &FFIQueueObj::clear)
      .def(
          "front",
          [](FFIQueueObj* self) {
            auto re = tvm::ffi::Any();
            auto fr = self->front();
            if (fr.has_value()) {
              re = fr.value();
            } 
            return re;
          })
    //   .def("try_pop", &FFIQueueObj::pop)
      .def(
          "push",
          static_cast<void (FFIQueueObj::*)(const tvm::ffi::Any&)>(
              &FFIQueueObj::push));
  ;
}

FFIQueue default_queue() {
  static FFIQueue s_default_queue;
  return s_default_queue;
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(default_queue, omniback::ffi::default_queue);

} // namespace

