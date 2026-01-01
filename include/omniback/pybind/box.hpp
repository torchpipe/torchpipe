#include <pybind11/pybind11.h>
#include "omniback/builtin/box.hpp"
#include "omniback/helper/omniback_export.h"
namespace omniback {

namespace py = pybind11;
// PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>);

// PYBIND11_DECLARE_HOLDER_TYPE(omniback::Boxes,
// std::shared_ptr<omniback::Boxes>);

OMNI_EXPORT void init_box(py::module_& m);
OMNI_EXPORT py::object cast(omniback::Boxes& boxes);

} // namespace omniback