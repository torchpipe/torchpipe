#include "py_register.hpp"

// #include <type_traits>
#include <typeinfo>
#include <unordered_map>
#include <functional>
#include <string>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "hami/core/any.hpp"
#include "hami/core/event.hpp"
#include "hami/helper/base_logging.hpp"

namespace hami::local {
using namespace hami;

std::unordered_map<std::type_index, any2obj_func>& get_type_map() {
    static std::unordered_map<std::type_index, any2obj_func> type_names;
    return type_names;
}

void try_insert(const std::type_index& type, const any2obj_func& func) {
    SPDLOG_INFO("register hash: {}. ", type.name());

    auto [it, inserted] = get_type_map().emplace(type, func);
    if (!inserted) {
        throw std::runtime_error("Type already registered");
    }
}

#define REGISTER_CONVERTER(type)               \
    g_type_names[typeid(type)] = any2obj_func( \
        [](const any& self) { return py::cast(any_cast<type>(self)); });

#define REGISTER_NO_VECTOR_CONVERTER(type)                               \
    g_type_names[typeid(type)] = any2obj_func(                           \
        [](const any& self) { return py::cast(any_cast<type>(self)); }); \
    g_type_names[typeid(std::unordered_set<type>)] =                     \
        any2obj_func([](const any& self) {                               \
            return py::cast(any_cast<std::unordered_set<type>>(self));   \
        });                                                              \
    g_type_names[typeid(std::unordered_map<std::string, type>)] =        \
        any2obj_func([](const any& self) {                               \
            return py::cast(                                             \
                any_cast<std::unordered_map<std::string, type>>(self));  \
        });

#define REGISTER_VECTOR_CONVERTER(type)                         \
    SPDLOG_DEBUG("REGISTER_VECTOR_CONVERTER {} ", #type);       \
    g_type_names[typeid(std::vector<type>)] =                   \
        any2obj_func([](const any& self) {                      \
            return py::cast(any_cast<std::vector<type>>(self)); \
        });

#define REGISTER_STRMAP_CONVERTER(type)                                 \
    g_type_names[typeid(std::unordered_map<std::string, type>)] =       \
        any2obj_func([](const any& self) {                              \
            return py::cast(                                            \
                any_cast<std::unordered_map<std::string, type>>(self)); \
        });

#define REGISTER_BYTES_CONVERTER(type)                                         \
    g_type_names[typeid(std::vector<type>)] =                                  \
        any2obj_func([](const any& self) {                                     \
            const std::vector<type>& data = any_cast<std::vector<type>>(self); \
            return py::bytes(reinterpret_cast<const char*>(data.data()),       \
                             data.size());                                     \
        });

#define REGISTER_LOCAL_TYPE_CONVERTER(type) \
    REGISTER_NO_VECTOR_CONVERTER(type)      \
    REGISTER_VECTOR_CONVERTER(type) REGISTER_STRMAP_CONVERTER(type)

namespace py = pybind11;
static auto _tmp = []() {
    static auto& g_type_names = get_type_map();
    REGISTER_LOCAL_TYPE_CONVERTER(int);
    REGISTER_LOCAL_TYPE_CONVERTER(float);
    REGISTER_LOCAL_TYPE_CONVERTER(double);
    REGISTER_LOCAL_TYPE_CONVERTER(bool);
    REGISTER_LOCAL_TYPE_CONVERTER(unsigned long);
    REGISTER_LOCAL_TYPE_CONVERTER(unsigned long long);
    REGISTER_LOCAL_TYPE_CONVERTER(long long);
    REGISTER_LOCAL_TYPE_CONVERTER(std::string);

    REGISTER_NO_VECTOR_CONVERTER(std::byte);
    REGISTER_BYTES_CONVERTER(std::byte);
    REGISTER_NO_VECTOR_CONVERTER(unsigned char);
    REGISTER_BYTES_CONVERTER(unsigned char);
    REGISTER_NO_VECTOR_CONVERTER(char);
    REGISTER_BYTES_CONVERTER(char);

    REGISTER_CONVERTER(std::shared_ptr<hami::Event>);
    return true;
}();

py::object any2object_from_hash_register(const any& input) {
    static auto& g_type_names = get_type_map();
    auto iter = g_type_names.find(input.type());
    if (iter != g_type_names.end()) {
        return iter->second(input);
    } else {
    }
    throw py::type_error("cpp2py: unregistered type: " +
                         std::string(input.type().name()));
    return py::object();
    // throw py::type_error("Can not convert any to python object");
}

// std::optional<any> object2any_base_type(const pybind11::object& data) {
//   if (py::isinstance<py::str>(data)) {
//     return py::cast<std::string>(data);
//   } else if (py::isinstance<py::int_>(data)) {
//     return py::cast<int>(data);
//   } else if (py::isinstance<py::float_>(data)) {
//     return py::cast<float>(data);
//   } else if (py::isinstance<Event>(data)) {
//     return py::cast<std::shared_ptr<Event>>(data);
//   } else if (py::isinstance<py::bytes>(data)) {
//     return py::cast<std::string>(data);
//   } else if (py::isinstance<TypedDict>(data)) {
//     return py::cast<std::shared_ptr<TypedDict>>(data);
//   } else if (py::isinstance<any>(data)) {
//     return py::cast<any>(data);
//   }
//   return std::nullopt;
// }
}  // namespace hami::local