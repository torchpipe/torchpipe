#pragma once

#include <pybind11/pybind11.h>

#include "hami/core/any.hpp"
#include "hami/helper/macro.h"
namespace hami::local {
namespace py = pybind11;
using any2obj_func = std::function<py::object(const any&)>;

py::object any2object_from_hash_register(const any&);
// HAMI_EXPORT std::unordered_map<std::type_index, any2obj_func>&
// get_type_map();
HAMI_EXPORT void try_insert(const std::type_index& type,
                            const any2obj_func& func);
template <typename T>
void register_any2object_hash_converter() {
    try_insert(typeid(T), any2obj_func([](const any& self) {
                   return py::cast(any_cast<T>(self));
               }));
    //   auto& type_map = get_type_map();
    //   auto [it, inserted] = type_map.emplace(
    //       typeid(T), any2obj_func([](const any& self) { return
    //       py::cast(any_cast<T>(self)); }));
    //   if (!inserted) {
    //     throw std::runtime_error("Type already registered");
    //   }
}
}  // namespace hami::local

namespace hami {
template <typename T>
class ConverterHashRegister {
   public:
    ConverterHashRegister() { local::register_any2object_hash_converter<T>(); }
};

}  // namespace hami
#define HAMI_ADD_HASH(T) \
    static hami::ConverterHashRegister<T> _hami_hash_cpp2py_static;
