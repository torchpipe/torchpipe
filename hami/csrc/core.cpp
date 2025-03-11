#include "hami/core/any.hpp"

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include "hami/core/task_keys.hpp"
#include "hami/csrc/py_register.hpp"

namespace hami {
namespace py = pybind11;

using namespace pybind11::literals;

void init_core(py::module_& m) {
    m.attr("TASK_RESULT_KEY") = py::cast(TASK_RESULT_KEY);
    m.attr("TASK_DATA_KEY") = py::cast(TASK_DATA_KEY);
    m.attr("TASK_BOX_KEY") = py::cast(TASK_BOX_KEY);
    m.attr("TASK_INFO_KEY") = py::cast(TASK_INFO_KEY);
    m.attr("TASK_NODE_NAME_KEY") = py::cast(TASK_NODE_NAME_KEY);
}
}  // namespace hami