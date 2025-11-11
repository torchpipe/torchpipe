#pragma once

#include <pybind11/pybind11.h>
#include <functional>
#include <optional>
#include <vector>
#include "omniback/core/any.hpp"
#include "omniback/csrc/py_register.hpp"
#include "omniback/helper/macro.h"

#define OMNI_MAKE_UNIQUE(base) OMNI_CONCAT(base, __LINE__)

namespace omniback {
namespace py = pybind11;

// Forward declaration of instance function
template <typename From, typename To>
class ConverterRegistryBase;

template <typename From, typename To>
OMNI_EXPORT ConverterRegistryBase<From, To>& converterRegistryInstance();

template <>
OMNI_EXPORT ConverterRegistryBase<pybind11::handle, any>&
converterRegistryInstance<pybind11::handle, any>();
template <>
OMNI_EXPORT ConverterRegistryBase<any, pybind11::object>&
converterRegistryInstance<any, pybind11::object>();
// Base class for converter registry
template <typename From, typename To>
class OMNI_EXPORT ConverterRegistryBase {
 public:
  using ConverterFunc = std::function<std::optional<To>(const From&)>;

  void add_converter(ConverterFunc converter) {
    converters_.push_back(std::move(converter));
  }

  const std::vector<ConverterFunc>& get_converters() const {
    return converters_;
  }

 protected:
  ConverterRegistryBase() = default;

 private:
  friend ConverterRegistryBase<From, To>& converterRegistryInstance<From, To>();

  std::vector<ConverterFunc> converters_;
};

// template <typename From, typename To>
// OMNI_EXPORT std::optional<To> convert(const From& from) {
//   std::optional<To> result;
//   for (const auto& converter : converterRegistryInstance<From,
//   To>().get_converters()) {
//     result = converter(from);
//     if (result) return result;
//   }
//   return std::nullopt;
// }

static inline std::optional<any> convert_py2any(const pybind11::handle& data) {
  std::optional<any> result;
  for (const auto& converter :
       converterRegistryInstance<pybind11::handle, any>().get_converters()) {
    result = converter(data);
    if (result)
      return result;
  }
  return std::nullopt;
}

// Converter register class
template <typename From, typename To>
class ConverterRegister {
 public:
  using ConverterFunc = typename ConverterRegistryBase<From, To>::ConverterFunc;

  explicit ConverterRegister(ConverterFunc converter) {
    converterRegistryInstance<From, To>().add_converter(std::move(converter));
  }

  // explicit ConverterRegister(std::type_index type_id) {
  //   converterRegistryInstance<From, To>().add_hash(type_id);
  // }

  ~ConverterRegister() = default;
};

} // namespace omniback

// Register Python to C++ converter
#define OMNI_ADD_PY2CPP(converter_func)                               \
  static omniback::ConverterRegister<pybind11::handle, omniback::any> \
  OMNI_MAKE_UNIQUE(_omniback_add_py2cpp_static_)(converter_func)

// Register C++ to Python converter
#define OMNI_ADD_CPP2PY(converter_func)                               \
  static omniback::ConverterRegister<omniback::any, pybind11::object> \
      _omniback_add_cpp2py_static_##__COUNTER__##_##__func__(converter_func)
