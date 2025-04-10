#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <pybind11/numpy.h>

#include "hami/core/any.hpp"
#include "hami/core/task_keys.hpp"
#include "hami/csrc/py_register.hpp"
#include "hami/helper/exception.hpp"
#include "hami/core/request_size.hpp"
#include "hami/core/event.hpp"
namespace py = pybind11;
using hami::error::ExceptionHolder;

using namespace pybind11::literals;
// 注册类型转换器
namespace pybind11::detail {
template <>
struct type_caster<std::exception_ptr> {
 public:
  PYBIND11_TYPE_CASTER(std::exception_ptr, _("ExceptionHolder"));

  // Python -> C++ 转换
  bool load(handle src, bool convert) {
    if (!src)
      return false;

    // 检查是否是ExceptionHolder实例
    auto holder_type = py::type::of<ExceptionHolder>();
    if (!py::isinstance(src, holder_type))
      return false;

    // 提取C++对象指针
    auto* holder = src.cast<ExceptionHolder*>();
    value = std::exception_ptr(*holder);
    return true;
  }

  // C++ -> Python 转换
  static handle cast(
      std::exception_ptr ptr,
      return_value_policy /* policy */,
      handle /* parent */) {
    return py::cast(ExceptionHolder(ptr)).release();
  }
};
} // namespace pybind11::detail

namespace hami {

void init_core(py::module_& m) {
  m.attr("TASK_RESULT_KEY") = py::cast(TASK_RESULT_KEY);
  m.attr("TASK_DATA_KEY") = py::cast(TASK_DATA_KEY);
  m.attr("TASK_BOX_KEY") = py::cast(TASK_BOX_KEY);
  m.attr("TASK_INFO_KEY") = py::cast(TASK_INFO_KEY);
  m.attr("TASK_NODE_NAME_KEY") = py::cast(TASK_NODE_NAME_KEY);
  m.attr("TASK_MSG_KEY") = py::cast(TASK_MSG_KEY);
  m.attr("TASK_REQUEST_ID_KEY") = py::cast(TASK_REQUEST_ID_KEY);
  m.attr("TASK_REQUEST_SIZE_KEY") = py::cast(TASK_REQUEST_SIZE_KEY);
  m.attr("TASK_EVENT_KEY") = py::cast(TASK_EVENT_KEY);

  py::class_<ExceptionHolder>(m, "ExceptionHolder")
      .def(py::init<std::exception_ptr>()) // 构造函数
      .def(
          "has_exception",
          &ExceptionHolder::has_exception) // 检查是否有异常
      .def("rethrow", &ExceptionHolder::rethrow);

  HAMI_ADD_HASH(std::exception_ptr);
  // ([](const any& data) {
  //     return py::cast(ExceptionHolder(any_cast<std::exception_ptr>(data)));
  // });

  // PYBIND11_MAKE_OPAQUE(std::vector<int>);
  // PYBIND11_MAKE_OPAQUE(std::vector<float>);
  // py::bind_vector<std::vector<int>>(m, "VectorInt");
  // py::bind_vector<std::vector<float>>(m, "VectorFloat");
}
} // namespace hami