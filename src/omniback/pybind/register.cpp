#include <pybind11/stl.h>

#include "omniback/core/dict.hpp"
#include "omniback/core/event.hpp"
#include "omniback/pybind/py_register.hpp"
#include "omniback/pybind/register.hpp"
#include "omniback/core/queue.hpp"

namespace omniback {

// 定义特化版本
template <>
OMNI_EXPORT ConverterRegistryBase<pybind11::handle, any>&
converterRegistryInstance<pybind11::handle, any>() {
  static ConverterRegistryBase<pybind11::handle, any> registry;
  return registry;
}

template <>
OMNI_EXPORT ConverterRegistryBase<any, pybind11::object>&
converterRegistryInstance<any, pybind11::object>() {
  static ConverterRegistryBase<any, pybind11::object> registry;
  return registry;
}

// OMNI_ADD_HASH(TypedDict);
OMNI_ADD_HASH(std::shared_ptr<TypedDict>);
// OMNI_ADD_HASH(std::shared_ptr<Queue>);

OMNI_ADD_HASH(std::shared_ptr<Event>);
OMNI_ADD_HASH(
    std::pair<std::unordered_map<std::string, std::string>, std::string>)(
    [](const any& data) {
      const auto* target = any_cast<
          std::pair<std::unordered_map<std::string, std::string>, std::string>>(
          &data);
      py::list result;
      result.append(py::cast(target->first));
      result.append(py::cast(target->second));
      return result;
    });
OMNI_ADD_HASH(
    std::pair<std::string, std::unordered_map<std::string, std::string>>)(
    [](const any& data) {
      const auto* target = any_cast<
          std::pair<std::string, std::unordered_map<std::string, std::string>>>(
          &data);
      py::list result;
      result.append(py::cast(target->first));
      result.append(py::cast(target->second));
      return result;
    });

} // namespace omniback