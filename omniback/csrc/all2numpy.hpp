#pragma once
#include <type_traits>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <vector>

namespace py = pybind11;

namespace omniback {

// Version for lvalues - makes a copy
template <typename T>
py::array_t<T> to_numpy(const std::vector<T>& vec) {
  auto result = py::array_t<T>(vec.size());
  py::buffer_info buf = result.request();
  std::copy(vec.begin(), vec.end(), static_cast<T*>(buf.ptr));
  return result;
}

template <typename T>
py::array_t<T> to_numpy(std::vector<T>&& vec) {
  auto vec_ptr = std::make_unique<std::vector<T>>(std::move(vec));

  const auto size = vec_ptr->size();
  T* data = vec_ptr->data();

  py::capsule capsule(vec_ptr.release(), [](void* v) {
    delete static_cast<std::vector<T>*>(v);
  });

  return py::array_t<T>(
      {size}, // shape
      {sizeof(T)}, // strides (可省略)
      data, // 数据指针
      capsule // 内存管理胶囊
  );
}

template <typename T>
py::array_t<T> to_numpy_with_no_gil(std::vector<T>&& vec) {
  auto vec_ptr = std::make_unique<std::vector<T>>(std::move(vec));

  const auto size = vec_ptr->size();
  T* data = vec_ptr->data();

  py::gil_scoped_acquire acquire;
  py::capsule capsule(vec_ptr.release(), [](void* v) {
    delete static_cast<std::vector<T>*>(v);
  });

  return py::array_t<T>(
      {size}, // shape
      {sizeof(T)}, // strides (可省略)
      data, // 数据指针
      capsule // 内存管理胶囊
  );
}

} // namespace omniback