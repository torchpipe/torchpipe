#include "omniback/csrc/py_helper.hpp"
#include "omniback/helper/base_logging.hpp"
#include "omniback/helper/macro.h"
namespace omniback::python {

size_t get_num_params(
    const py::object& obj,
    const char* method,
    size_t* defaults_count) {
  py::object method_attr = obj.attr(method);
  OMNI_ASSERT(py::isinstance<py::function>(method_attr));
  // SPDLOG_INFO(
  //     "__func__ {} __code__ {}",
  //     py::hasattr(obj, "__func__"),
  //     py::hasattr(obj, "__code__"));
  if (!py::hasattr(method_attr, "__code__")) {
    return 0;
  }
  // OMNI_ASSERT(
  //     py::hasattr(method_attr, "__code__"),
  //     "no __code__ for " + std::string(method));
  // if (!py::hasattr(obj, "__code__")) {
  //   return 1;
  // }
  auto code_obj = method_attr.attr("__code__");
  int arg_count = code_obj.attr("co_argcount").cast<int>();

  if (defaults_count != nullptr) {
    py::object defaults = method_attr.attr("__defaults__");
    // 检查是否有默认值
    if (!defaults.is_none()) {
      auto defaults_tuple = defaults.cast<py::tuple>();
      *defaults_count = defaults_tuple.size();
      SPDLOG_DEBUG("Number of default arguments: {}", *defaults_count);

      for (int i = 0; i < *defaults_count; ++i) {
        SPDLOG_DEBUG(
            "Default value for argument {}: {}",
            i,
            py::cast<std::string>(py::str(defaults_tuple[i])));
      }
    }
  }
  return arg_count;
}
} // namespace omniback::python