

#pragma once

#include <pybind11/pybind11.h>
#include <cstddef>
#include <memory>

namespace omniback {
namespace python {
namespace py = pybind11;

template <typename T>
struct Pybind11Deleter {
  void operator()(T* obj) const {
    if (obj) {
      py::gil_scoped_acquire acquire;
      delete obj;
    }
  }
};

template <typename T>
auto make_unique(const T& obj) {
  return std::unique_ptr<T, Pybind11Deleter<T>>(new T(obj));
}

template <typename T>
auto make_shared(const T& obj) {
  return std::shared_ptr<T>(new T(obj), Pybind11Deleter<T>{});
}

// template <typename T, typename... Args>
// auto make_unique(Args&&... args) {
//   return std::unique_ptr<T, Pybind11Deleter<T>>{
//       new T(std::forward<Args>(args)...), Pybind11Deleter<T>()};
// }
template <typename T = py::object>
using unique_ptr = std::unique_ptr<T, Pybind11Deleter<T>>;

size_t get_num_params(
    const py::object& obj,
    const char* method,
    size_t* defaults_count = nullptr);

} // namespace python
} // namespace omniback