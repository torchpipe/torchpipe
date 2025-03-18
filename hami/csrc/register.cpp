#include <pybind11/stl.h>

#include "hami/csrc/register.hpp"
#include "hami/csrc/py_register.hpp"
#include "hami/core/dict.hpp"

namespace hami {

// template <typename From, typename To>
// HAMI_EXPORT ConverterRegistryBase<From, To>& converterRegistryInstance() {
//   static ConverterRegistryBase<From, To> registry;
//   return registry;
// }

// template class HAMI_EXPORT ConverterRegistryBase<pybind11::handle, any>&
// converterRegistryInstance<pybind11::handle, any>();
// template class HAMI_EXPORT ConverterRegistryBase<any, pybind11::object>&
// converterRegistryInstance<any, pybind11::object>();

// 定义特化版本
template <>
HAMI_EXPORT ConverterRegistryBase<pybind11::handle, any> &
converterRegistryInstance<pybind11::handle, any>() {
    static ConverterRegistryBase<pybind11::handle, any> registry;
    return registry;
}

template <>
HAMI_EXPORT ConverterRegistryBase<any, pybind11::object> &
converterRegistryInstance<any, pybind11::object>() {
    static ConverterRegistryBase<any, pybind11::object> registry;
    return registry;
}

HAMI_ADD_HASH(TypedDict);
HAMI_ADD_HASH(std::pair<std::unordered_map<std::string, std::string>,
                        std::string>)([](const any &data) {
    const auto *target = any_cast<
        std::pair<std::unordered_map<std::string, std::string>, std::string>>(
        &data);
    py::list result;
    result.append(py::cast(target->first));
    result.append(py::cast(target->second));
    return result;
});
HAMI_ADD_HASH(
    std::pair<std::string, std::unordered_map<std::string, std::string>>)(
    [](const any &data) {
        const auto *target = any_cast<std::pair<
            std::string, std::unordered_map<std::string, std::string>>>(&data);
        py::list result;
        result.append(py::cast(target->first));
        result.append(py::cast(target->second));
        return result;
    });

}  // namespace hami