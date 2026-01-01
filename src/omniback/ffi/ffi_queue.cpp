#include <unordered_map>
#include <mutex>

#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/function.h>
#include <omniback/ffi/queue.h>
#include <tvm/ffi/reflection/registry.h>
// #include <tvm/ffi/reflection/overload.h>
#include <tvm/ffi/base_details.h>
#include <tvm/ffi/extra/stl.h>

#include "omniback/ffi/dict.h"

namespace omniback::ffi {

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  using omniback::ffi::Any;

  refl::ObjectDef<ThreadSafeQueueObj>()
      // Constructors
      .def(refl::init<>())
      .def(
          "size",
          [](const ThreadSafeQueueObj* obj) {
            return static_cast<int64_t>(obj->size());
          })
      .def("empty", &ThreadSafeQueueObj::empty)
      .def("clear", &ThreadSafeQueueObj::clear)
      .def("front_size", &ThreadSafeQueueObj::front_size)

      // Capacity management
      .def("max_size", &ThreadSafeQueueObj::max_size)
      .def("set_max_size", &ThreadSafeQueueObj::set_max_size)

      // Waiting operations
      .def(
          "wait_for",
          [](ThreadSafeQueueObj* obj, size_t timeout_ms) {
            return obj->wait_for(timeout_ms);
          })
      // Element access (non-blocking front access)
      .def(
          "front",
          [](const ThreadSafeQueueObj* obj) { return obj->front<Any>(); })
      // Push operations
      .def(
          "put",
          [](ThreadSafeQueueObj* obj, const tvm::ffi::Any& value) {
            obj->push(value);
          })
      .def(
          "put_with_size",
          [](ThreadSafeQueueObj* obj,
             const tvm::ffi::Any& value,
             size_t real_size) { obj->push_with_size<Any>(value, real_size); })
      .def(
          "wait_until_at_least",
          [](ThreadSafeQueueObj* obj, size_t queue_size, size_t timeout_ms) {
            obj->wait_until_at_least(queue_size, timeout_ms);
          })
      .def(
          "put_with_max_limit",
          [](ThreadSafeQueueObj* obj,
             const tvm::ffi::Any& value,
             size_t max_size,
             size_t timeout_ms) {
            return obj->push_with_max_limit<Any>(value, max_size, timeout_ms);
          })

      // Get operations
      .def(
          "get",
          [](ThreadSafeQueueObj* obj, bool block) -> Any {
            if (!block)
              return obj->get<Any>();
            else{
              return obj->wait_get<Any>();
            }
          })
      .def("try_get", [](ThreadSafeQueueObj* obj, size_t timeout_ms) {
        return obj->try_get<Any>(timeout_ms);
      });
}

static std::unordered_map<std::string, ThreadSafeQueueRef>& GetQueueRegistry() {
  static std::unordered_map<std::string, ThreadSafeQueueRef> registry;
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
    return *(it->second.get());
  }

  ThreadSafeQueueRef q =
      ThreadSafeQueueRef(tvm::ffi::make_object<ThreadSafeQueueObj>());
  registry[tag] = q;
  return *(registry[tag].get());
}

ThreadSafeQueueObj* py_default_queue_one_arg(const std::string& tag = ""){
  auto& q = default_queue(tag);
  return &q;
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(default_queue_one_arg, py_default_queue_one_arg);

} // namespace omniback::ffi
