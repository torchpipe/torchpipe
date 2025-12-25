#include <unordered_map>
#include <mutex>

#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/function.h>
#include <omniback/ffi/queue.h>
#include <tvm/ffi/reflection/registry.h>

namespace omniback::ffi {


TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<ThreadSafeQueueObj>()
      .def(refl::init<>())
      .def("size", &ThreadSafeQueueObj::size)
      .def("empty", &ThreadSafeQueueObj::empty)
      .def("clear", &ThreadSafeQueueObj::clear)
      .def("front", &ThreadSafeQueueObj::front)
      .def("get", &ThreadSafeQueueObj::get)
      .def(
          "put",
          static_cast<void (ThreadSafeQueueObj::*)(const tvm::ffi::Any&)>(
              &ThreadSafeQueueObj::put));
  ;
}

static std::unordered_map<std::string, Queue>& GetQueueRegistry() {
  static std::unordered_map<std::string, Queue> registry;
  return registry;
}

static std::mutex& GetQueueRegistryMutex() {
  static std::mutex m;
  return m;
}

void cleanup_queue(){
  std::lock_guard<std::mutex> lock(GetQueueRegistryMutex());
  auto& registry = GetQueueRegistry();
  registry.clear();
}

ThreadSafeQueueObj& default_queue(const std::string& tag) {
  std::lock_guard<std::mutex> lock(GetQueueRegistryMutex());
  auto& registry = GetQueueRegistry();

  auto it = registry.find(tag);
  if (it != registry.end()) {
    return it->second;
  }

  Queue q(tvm::ffi::make_object<ThreadSafeQueueObj>());
  registry[tag] = q;
  return registry[tag];
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(default_queue_one_arg, omniback::ffi::default_queue);

} // namespace omniback::ffi
