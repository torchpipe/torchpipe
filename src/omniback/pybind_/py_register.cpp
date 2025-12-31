#include "omniback/pybind/py_register.hpp"

// #include <type_traits>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <functional>
#include <string>
#include <typeinfo>
#include <unordered_map>

#include "omniback/core/any.hpp"
#include "omniback/core/event.hpp"
#include "omniback/helper/base_logging.hpp"

namespace omniback::reg {
using namespace omniback;

std::unordered_map<size_t, std::pair<any2obj_func, obj2any_func>>&
get_type_map() {
  static std::unordered_map<size_t, std::pair<any2obj_func, obj2any_func>>
      type_names;
  return type_names;
}

void try_insert(
    const std::type_info& type,
    const any2obj_func& func,
    const obj2any_func& o2a) {
  // SPDLOG_INFO("register hash: {}. ", type.name());

  auto [it, inserted] = get_type_map().emplace(
      type.hash_code(), std::pair<any2obj_func, obj2any_func>{func, o2a});
  if (!inserted) {
    throw std::runtime_error(
        "Type already registered: " + std::string(type.name()));
  }
}

#define REGISTER_CONVERTER(type) register_any_object_hash_converter<type>();

#define REGISTER_NO_VECTOR_CONVERTER(type)                        \
  register_any_object_hash_converter<type>();                     \
  register_any_object_hash_converter<std::unordered_set<type>>(); \
  register_any_object_hash_converter<std::unordered_map<std::string, type>>();

#define REGISTER_VECTOR_CONVERTER(type)                 \
  SPDLOG_DEBUG("REGISTER_VECTOR_CONVERTER {} ", #type); \
  register_any_object_hash_converter<std::vector<type>>(); \
  register_any_object_hash_converter<std::vector<std::vector<type>>>();

#define REGISTER_STRMAP_CONVERTER(type) \
  register_any_object_hash_converter<std::unordered_map<std::string, type>>();

#define REGISTER_BYTES_CONVERTER(type) \
  register_any_object_hash_converter<std::vector<type>>();

#define REGISTER_LOCAL_TYPE_CONVERTER(type) \
  REGISTER_NO_VECTOR_CONVERTER(type)        \
  REGISTER_VECTOR_CONVERTER(type)

namespace py = pybind11;
static auto _tmp = []() {
  // static auto &g_type_names = get_type_map();
  REGISTER_LOCAL_TYPE_CONVERTER(int);
  REGISTER_LOCAL_TYPE_CONVERTER(float);
  REGISTER_LOCAL_TYPE_CONVERTER(double);
  REGISTER_LOCAL_TYPE_CONVERTER(bool);
  // REGISTER_LOCAL_TYPE_CONVERTER(unsigned long);
  // REGISTER_LOCAL_TYPE_CONVERTER(unsigned long long);
  // REGISTER_LOCAL_TYPE_CONVERTER(long long);
  REGISTER_LOCAL_TYPE_CONVERTER(std::string);

  REGISTER_NO_VECTOR_CONVERTER(std::byte);
  REGISTER_BYTES_CONVERTER(std::byte);
  REGISTER_NO_VECTOR_CONVERTER(unsigned char);
  REGISTER_BYTES_CONVERTER(unsigned char);
  REGISTER_NO_VECTOR_CONVERTER(char);
  REGISTER_BYTES_CONVERTER(char);

  // REGISTER_CONVERTER(std::shared_ptr<omniback::Event>);
  return true;
}();

py::object any2object_from_hash_register(const any& input) {
  static auto& g_type_names = get_type_map();
  
  auto iter = g_type_names.find(input.type().hash_code());
  if (iter != g_type_names.end()) {
    // if (input.type() == typeid(std::string))
    // {
    //     std::string re = any_cast<std::string>(input);
    //     SPDLOG_INFO(" {} ", re);
    // }
    // SPDLOG_INFO("input.type().name {} ", input.type().name());
    // OMNI_ASSERT(iter->second.first);
    return iter->second.first(input);
  } else {
    return py::cast(input);
  }
  throw py::type_error(
      "cpp2py: unregistered type: " + std::string(input.type().name()));
  return py::object();
  // throw py::type_error("Can not convert any to python object");
}

std::optional<any> object2any_from_hash_register(const py::handle& input) {
  // const auto cls = py::type::handle_of(input.get_type());
  if (py::hasattr(input, "type_hash")) {
    // throw std::runtime_error("type_hash test");
    const static auto& g_type_names = get_type_map();
    size_t hash = py::cast<size_t>(input.attr("type_hash"));
    auto iter = g_type_names.find(hash);
    if (iter != g_type_names.end()) {
      return iter->second.second(handle2object(input));
    }
  }
  return std::nullopt;
}

// std::optional<any> object2any_base_type(const pybind11::object& data) {
//   if (py::isinstance<py::str>(data)) {
//     return py::cast<std::string>(data);
//   } else if (py::isinstance<py::int_>(data)) {
//     return py::cast<int>(data);
//   } else if (py::isinstance<py::float_>(data)) {
//     return py::cast<float>(data);
//   } else if (py::isinstance<Event>(data)) {
//     return py::cast<Event>(data);
//   } else if (py::isinstance<py::bytes>(data)) {
//     return py::cast<std::string>(data);
//   } else if (py::isinstance<TypedDict>(data)) {
//     return py::cast<std::shared_ptr<TypedDict>>(data);
//   } else if (py::isinstance<any>(data)) {
//     return py::cast<any>(data);
//   }
//   return std::nullopt;
// }
} // namespace omniback::reg