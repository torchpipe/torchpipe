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

bool use_cxx11_abi() {
#if USE_CXX11_ABI
  return true;
#else
  return false;
#endif
}

PYBIND11_MODULE(PY_MODULE_NAME, m) {
  m.doc() = "omniback C++ extension";
  m.def("get_version", &get_version);
  m.def("use_cxx11_abi", &use_cxx11_abi);
  m.def("_GLIBCXX_USE_CXX11_ABI", &use_cxx11_abi);

  init_any(m);
  init_dict(m);
  init_event(m);

  py_init_backend(m);

  init_core(m);
  init_box(m);
}

} // namespace omniback