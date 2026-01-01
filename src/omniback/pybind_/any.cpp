

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include "omniback/core/any.hpp"
#include "omniback/core/event.hpp"
#include "omniback/pybind/any.hpp"
#include "omniback/pybind/converts.hpp"
#include "omniback/helper/string.hpp"

namespace omniback {

constexpr auto set_double_error_message() {
  return "The input data is a floating-point value, but it exceeds the "
         "representable range of a "
         "C++ float. We currently do not allow automatic conversion to "
         "double. If it is "
         "necessary, please construct a default any and then call its "
         "set_double function.";
}
constexpr auto set_size_t_error_message() {
  return "The input data is a int value, but it exceeds the representable "
         "range of a "
         "C++ int. We currently do not allow automatic conversion to size_t. "
         "If it is "
         "necessary, please construct a default any and then call its "
         "set_size_t function.";
}

#define DEFINE_CONVERSION_AS(type, name) \
  def("as_" #name, [](const any& self) { return any_cast<type>(self); })

#define DEFINE_CONVERSION_FUNCTIONS(type, name)                               \
  def(py::init<type>())                                                       \
      .def("as_" #name, [](const any& self) { return any_cast<type>(self); }) \
      .def(py::init<std::vector<type>>())                                     \
      .def(                                                                   \
          "as_list_of_" #name,                                                \
          [](const any& self) { return any_cast<std::vector<type>>(self); })  \
      .def(py::init<std::unordered_set<type>>())                              \
      .def(                                                                   \
          "as_set_of_" #name,                                                 \
          [](const any& self) {                                               \
            return any_cast<std::unordered_set<type>>(self);                  \
          })                                                                  \
      .def(py::init<std::unordered_map<std::string, type>>())                 \
      .def("as_dict_of_" #name, [](const any& self) {                         \
        return any_cast<std::unordered_map<std::string, type>>(self);         \
      })

namespace py = pybind11;
using namespace pybind11::literals;

void init_any(py::module_& m) {
  py::class_<omniback::any, std::shared_ptr<omniback::any>> omniback_any(
      m, "Any");

  omniback_any.doc() =
      "omniback.Any provides object wrapper for "
      "omniback::any class. It allows to pass different types of objects"
      "into C++ based core of the project.";
  omniback_any
      .def(py::init([](py::object& obj) {
        auto re = object2any(obj);
        if (re == std::nullopt) {
          throw py::type_error(
              "The input data is not supported by omniback.Any.");
        }
        return *re;
      }))
      .def_property_readonly(
          "value", [](const any& self) { return any2object(self); })
      .def("cast", [](const any& self) { return any2object(self); })
      .DEFINE_CONVERSION_AS(std::string, str)
      .DEFINE_CONVERSION_AS(int, int)
      .DEFINE_CONVERSION_AS(size_t, size_t)
      .DEFINE_CONVERSION_AS(float, float)
      .DEFINE_CONVERSION_AS(double, double)
      .DEFINE_CONVERSION_AS(bool, bool)
      .DEFINE_CONVERSION_AS(std::byte, byte)
      // .DEFINE_CONVERSION_AS(std::vector<std::byte>, bytes)
      .def(
          "as_bytes",
          [](const any& self) {
            if (self.type() == typeid(std::vector<unsigned char>)) {
              const auto& data = any_cast<std::vector<unsigned char>>(self);
              const char* data_ptr = reinterpret_cast<const char*>(data.data());
              return py::bytes(data_ptr, data.size());
            } else if (self.type() == typeid(std::vector<char>)) {
              const auto& data = any_cast<std::vector<char>>(self);
              const char* data_ptr = reinterpret_cast<const char*>(data.data());
              return py::bytes(data_ptr, data.size());
            } else if (self.type() == typeid(std::vector<std::byte>)) {
              const auto& data = any_cast<std::vector<std::byte>>(self);
              const char* data_ptr = reinterpret_cast<const char*>(data.data());
              return py::bytes(data_ptr, data.size());
            } else if (self.type() == typeid(std::string)) {
              const auto& data = any_cast<std::string>(self);
              const char* data_ptr = reinterpret_cast<const char*>(data.data());
              return py::bytes(data_ptr, data.size());
            }
            throw py::type_error(
                "The input data cannot be interpreted as bytes. Supported "
                "types: vector<byte>, "
                "vector<unsigned char>, vector<char>, string.");
          })
      // .DEFINE_CONVERSION_AS(TypedDict, typed_dict)
      .DEFINE_CONVERSION_AS(Event, event);
}

} // namespace omniback
