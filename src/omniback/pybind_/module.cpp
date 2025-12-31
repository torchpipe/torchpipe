#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace omniback {
namespace py = pybind11;

extern void init_core(py::module_& m);
extern void init_any(py::module_& m);
extern void init_dict(py::module_& m);
extern void py_init_backend(py::module_& m);
extern void init_event(py::module_& m);
extern void init_box(py::module_& m);

class CZZ {};
// NOLINTNEXTLINE
static py::object get_version() {
  CZZ zz;
  try {
    return py::cast(zz);
  } catch (py::error_already_set& e) {
    return py::cast(2);
  }
}

void translate_python_error() {
  if (PyErr_Occurred()) {
    py::error_already_set e;
    throw std::runtime_error(e.what());
  }
}


PYBIND11_MODULE(PY_MODULE_NAME, m) {
  m.doc() = "omniback C++ extension";

  py::register_exception_translator([](std::exception_ptr p) {
    try {
      if (p)
        std::rethrow_exception(p);
    } catch (const py::error_already_set& e) {
      translate_python_error();
    }
  });
  
  m.def("get_version", &get_version);

  init_any(m);
  init_dict(m);
  init_event(m);

  py_init_backend(m);

  init_core(m);
  init_box(m);
}

} // namespace omniback