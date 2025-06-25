#include "hami/builtin/box.hpp"
#include <pybind11/pybind11.h>
#include "hami/helper/hami_export.h"
namespace hami {

namespace py = pybind11;
// PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>);

// PYBIND11_DECLARE_HOLDER_TYPE(hami::Boxes, std::shared_ptr<hami::Boxes>);

HAMI_EXPORT void init_box(py::module_& m);
HAMI_EXPORT py::object cast(hami::Boxes& boxes);

} // namespace hami