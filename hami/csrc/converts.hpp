#pragma once

#include <string>
#include <optional>
#include <pybind11/pybind11.h>

#include "hami/core/any.hpp"
namespace hami {
constexpr auto EMPTY_DICT_NOT_CONVERTABLE =
    "Cannot convert empty dict to C++ type due to ambiguous type information. "
    "If needed, consider using specific types like hami.str_map.";
constexpr auto UNKNOWN_PYTHON_TYPE =
    "You are trying to convert an unsupported Python type to a C++ type. Please ensure that you "
    "have used "
    "`register_py2cpp(std::function<std::optional<hami::any>(const pybind11::handle&))` to "
    "register the conversion function";
namespace py = pybind11;

std::optional<any> object2any(const py::handle&);
py::object any2object(const any& input);
void add_object2any_converter(std::function<any(py::object)>);

void add_any2object_converter(std::function<py::object(const any&)>);

// void register_py2cpp(std::function < std::optional<hami::any>(const pybind11::handle&));

template <typename keyT, typename valueT>
std::unordered_map<keyT, valueT> convert_dict(py::dict d) {
  std::unordered_map<keyT, valueT> result;
  for (const auto& item : d) {
    auto key = item.first.cast<keyT>();
    auto value = item.second.cast<valueT>();
    result[key] = value;
  }
  return result;
}

template <typename T>
std::vector<T> convert_list(py::list d) {
  std::vector<T> result;
  for (const auto& item : d) {
    result.push_back(item.cast<T>());
  }
  return result;
}

// void HAMI_EXPORT
// register_py2cpp(std::function<std::optional<any>(const pybind11::handle&)> converter);
// void HAMI_EXPORT
// register_cpp2py(std::function<std::optional<pybind11::object>(const hami::any&)> converter);
// const std::vector<std::function<std::optional<pybind11::object>(const hami::any&)>>&
// get_cpp2py_registers();
// const std::vector<std::function<std::optional<hami::any>(const pybind11::handle&)>>&
// get_py2cpp_registers();
}  // namespace hami