#include "hami/csrc/py_helper.hpp"
#include "hami/helper/base_logging.hpp"
#include "hami/helper/macro.h"
namespace hami::python {

size_t get_num_params(const py::object& obj, const char* method, size_t* defaults_count) {
  py::object method_attr = obj.attr(method);
  HAMI_ASSERT(py::isinstance<py::function>(method_attr));
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
        SPDLOG_DEBUG("Default value for argument {}: {}", i,
                     py::cast<std::string>(py::str(defaults_tuple[i])));
      }
    }
  }
  return arg_count;
}
}  // namespace hami::python