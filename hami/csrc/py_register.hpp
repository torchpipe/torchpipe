#pragma once

#include <pybind11/pybind11.h>

#include "hami/core/any.hpp"
#include "hami/helper/macro.h"
namespace hami::reg {
namespace py = pybind11;
using any2obj_func = std::function<py::object(const any &)>;
using obj2any_func = std::function<any(const py::object &)>;

py::object any2object_from_hash_register(const any &);
std::optional<any> object2any_from_hash_register(const py::handle &);

inline py::object handle2object(const py::handle &han) {
    return py::reinterpret_borrow<py::object>(han);
}
// HAMI_EXPORT std::unordered_map<std::type_index, any2obj_func>&
// get_type_map();
HAMI_EXPORT void try_insert(const std::type_info &type,
                            const any2obj_func &func = nullptr,
                            const obj2any_func &func2 = nullptr);
template <typename T>
void register_any_object_hash_converter(any2obj_func conv = nullptr,
                                        obj2any_func conv2 = nullptr) {
    if (!conv)
        conv = [](const any &self) { return py::cast(any_cast<T>(self)); };
    if (!conv2)
        conv2 = [](const py::object &self) { return (any)py::cast<T>(self); };

    try_insert(typeid(T), conv, conv2);
}

template <typename T>
void register_any_ptr_object_hash_converter(any2obj_func conv = nullptr,
                                            obj2any_func conv2 = nullptr) {
    // if (!conv)
    //     conv = [](const any &self) { return py::cast(any_cast<T>(self)); };
    if (!conv2)
        conv2 = [](const py::object &self) { return (any)py::cast<T *>(self); };

    try_insert(typeid(T), conv, conv2);
}

}  // namespace hami::reg

namespace hami::reg {
template <typename T>
class ConverterHashRegister {
   public:
    ConverterHashRegister() { register_any_object_hash_converter<T>(); }
    ConverterHashRegister(any2obj_func func) {
        register_any_object_hash_converter<T>(func);
    }
};

}  // namespace hami::reg

// 参数包装宏（处理含逗号的模板参数）
#define HAMI_ARG(...) __VA_ARGS__

#define HAMI_CONCAT_IMPL(a, b) a##b
#define HAMI_CONCAT(a, b) HAMI_CONCAT_IMPL(a, b)

#define HAMI_MAKE_UNIQUE_SUFFIX() \
    HAMI_CONCAT(_hami_hash_, HAMI_CONCAT(__LINE__, HAMI_CONCAT(_, __COUNTER__)))

#define HAMI_ADD_HASH(...)                                            \
    static hami::reg::ConverterHashRegister<__VA_ARGS__> HAMI_CONCAT( \
        d_, HAMI_MAKE_UNIQUE_SUFFIX())

// #define HAMI_ADD_HASH(T)   static hami::ConverterHashRegister<T>
// _hami_hash_cpp2py_static_##__COUNTER__;
