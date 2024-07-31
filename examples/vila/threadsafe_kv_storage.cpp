#include <optional>

#include "threadsafe_kv_storage.hpp"
#include "any2object.hpp"
#include "object2any.hpp"

#include "threadsafe_queue.hpp"

namespace ipipe {

// 获取单例实例
ThreadSafeKVStorage& ThreadSafeKVStorage::getInstance() {
  static ThreadSafeKVStorage instance;
  return instance;
}

// 读取数据
std::optional<ipipe::any> ThreadSafeKVStorage::read(const std::string& path) {
  std::shared_lock<std::shared_mutex> lock(mutex_);
  auto it = disk_.find(path);
  if (it != disk_.end()) {
    return it->second;
  }
  return std::nullopt;  // 返回空值表示未找到
}

// 写入数据
void ThreadSafeKVStorage::write(const std::string& path, ipipe::any data) {
  std::unique_lock<std::shared_mutex> lock(mutex_);
  disk_[path] = data;
}

void ThreadSafeKVStorage::clear() {
  std::unique_lock<std::shared_mutex> lock(mutex_);
  disk_.clear();
}

void ThreadSafeKVStorage::clear(const std::string& path) {
  std::unique_lock<std::shared_mutex> lock(mutex_);
  auto iter = disk_.find(path);
  if (iter != disk_.end()) {
    disk_.erase(iter);
  }
  throw std::invalid_argument("can not found key: " + path);
}

}  // namespace ipipe
#ifdef PYBIND
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "tensor_type_caster.hpp"
#endif

#ifdef PYBIND
namespace ipipe {
namespace py = pybind11;

template <typename T>
void bind_threadsafe_queue(py::module& m, const std::string& typestr) {
  using Queue = ipipe::ThreadSafeQueue<T>;
  std::string pyclass_name = std::string("ThreadSafeQueue") + typestr;
  py::class_<Queue, std::shared_ptr<Queue>>(m, pyclass_name.c_str())
      .def(py::init<>())
      .def("Push", py::overload_cast<const T&>(&Queue::Push),
           py::call_guard<py::gil_scoped_release>())
      .def("Push", py::overload_cast<const std::vector<T>&>(&Queue::Push),
           py::call_guard<py::gil_scoped_release>())
      .def(
          "WaitForPop",
          [](Queue& q, int time_out) -> py::object {
            T value;
            bool result = q.WaitForPop(value, time_out);
            py::gil_scoped_acquire local_guard;
            if (result) {
              return py::cast(value);
            } else {
              return py::none();
            }
          },
          py::call_guard<py::gil_scoped_release>())
      .def("empty", &Queue::empty, py::call_guard<py::gil_scoped_release>())
      .def("size", &Queue::size, py::call_guard<py::gil_scoped_release>());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  bind_threadsafe_queue<int>(m, "Int");
  bind_threadsafe_queue<float>(m, "Float");
  bind_threadsafe_queue<double>(m, "Double");
  bind_threadsafe_queue<long>(m, "Long");

  py::class_<ThreadSafeKVStorage>(m, "ThreadSafeKVStorage")
      .def("read",
           [](ThreadSafeKVStorage& self, const std::string& path) {
             std::optional<ipipe::any> result;
             {
               py::gil_scoped_release local_guard;
               result = self.read(path);
             }

             if (result == std::nullopt) {
               return py::object(py::none());
             } else
               return ipipe::any2object(*result);
             //  return ipipe::any2object(result);
           })
      .def("write", [](ThreadSafeKVStorage& self, const std::string& path,
                       pybind11::handle data) { self.write(path, object2any(data)); })
      .def("clear", py::overload_cast<>(&ThreadSafeKVStorage::clear))
      .def("clear", py::overload_cast<const std::string&>(&ThreadSafeKVStorage::clear))
      .def_static("getInstance", &ThreadSafeKVStorage::getInstance,
                  py::return_value_policy::reference);
}
}  // namespace ipipe
#endif