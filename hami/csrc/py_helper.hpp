

#pragma once

#include <memory>
#include <cstddef>
#include <pybind11/pybind11.h>

namespace hami {
namespace python {
namespace py = pybind11;

struct PyObjectDeleter {
  void operator()(py::object* obj) const {
    py::gil_scoped_acquire acquire;
    delete obj;
  }
};

// Define a type alias for the smart pointer with custom deleter
using unique_ptr = std::unique_ptr<py::object, PyObjectDeleter>;

static inline unique_ptr make_unique(const py::object& obj) {
  return unique_ptr(new py::object(obj));
}

size_t get_num_params(const py::object& obj, const char* method, size_t* defaults_count = nullptr);

}  // namespace python
}  // namespace hami