#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include "omniback/core/any.hpp"
#include "omniback/core/event.hpp"
#include "omniback/core/request_size.hpp"
#include "omniback/core/task_keys.hpp"
#include "omniback/csrc/atomic_type.hpp"
#include "omniback/csrc/py_register.hpp"
#include "omniback/helper/exception.hpp"

namespace py = pybind11;
using omniback::error::ExceptionHolder;

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

namespace omniback {

void init_core(py::module_& m) {
  m.attr("TASK_RESULT_KEY") = py::cast(TASK_RESULT_KEY);
  m.attr("TASK_DATA_KEY") = py::cast(TASK_DATA_KEY);
  m.attr("TASK_BOX_KEY") = py::cast(TASK_BOX_KEY);
  m.attr("TASK_INFO_KEY") = py::cast(TASK_INFO_KEY);
  m.attr("TASK_NODE_NAME_KEY") = py::cast(TASK_NODE_NAME_KEY);
  m.attr("TASK_MSG_KEY") = py::cast(TASK_MSG_KEY);
  m.attr("TASK_REQUEST_ID_KEY") = py::cast(TASK_REQUEST_ID_KEY);
  m.attr("TASK_WAITING_EVENT_KEY") = py::cast(TASK_WAITING_EVENT_KEY);
  m.attr("TASK_REQUEST_SIZE_KEY") = py::cast(TASK_REQUEST_SIZE_KEY);
  m.attr("TASK_EVENT_KEY") = py::cast(TASK_EVENT_KEY);

  py::class_<ExceptionHolder>(m, "ExceptionHolder")
      .def(py::init<std::exception_ptr>()) // 构造函数
      .def(
          "has_exception",
          &ExceptionHolder::has_exception) // 检查是否有异常
      .def("rethrow", &ExceptionHolder::rethrow);

  OMNI_ADD_HASH(std::exception_ptr);
  OMNI_ADD_HASH(std::pair<int, int>);
  OMNI_ADD_HASH(long long);
  OMNI_ADD_HASH(std::pair<float, float>);
  OMNI_ADD_HASH(std::pair<std::string, std::string>);

  py::class_<AtomicType<int>>(m, "AtomicInt")
      .def(py::init<int>(), py::arg("value") = 0)
      .def("get", &AtomicType<int>::get)
      .def("set", &AtomicType<int>::set)
      // 正确绑定 += 操作符
      .def(
          "__iadd__",
          &AtomicType<int>::operator+=,
          py::return_value_policy::reference_internal,
          py::arg("delta"))
      // 明确区分前缀/后缀 ++
      .def(
          "increment",
          [](AtomicType<int>& self) -> int { return (++self); },
          py::return_value_policy::reference_internal)
      .def(
          "postfix_increment",
          [](AtomicType<int>& self) -> int { return self++; })
      .def(
          "compare_exchange",
          &AtomicType<int>::compare_exchange,
          py::arg("expected"),
          py::arg("desired"));
}
} // namespace omniback