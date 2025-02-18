#pragma once

#include <pybind11/pybind11.h>
#include "hami/core/any.hpp"
#include <vector>
#include <functional>
#include <optional>

namespace hami {
namespace py = pybind11;

// Forward declaration of instance function
template <typename From, typename To>
class ConverterRegistryBase;

template <typename From, typename To>
HAMI_EXPORT ConverterRegistryBase<From, To>& converterRegistryInstance();

template <>
HAMI_EXPORT ConverterRegistryBase<pybind11::handle, any>&
converterRegistryInstance<pybind11::handle, any>();
template <>
HAMI_EXPORT ConverterRegistryBase<any, pybind11::object>&
converterRegistryInstance<any, pybind11::object>();
// Base class for converter registry
template <typename From, typename To>
class HAMI_EXPORT ConverterRegistryBase {
 public:
  using ConverterFunc = std::function<std::optional<To>(const From&)>;

  void add_converter(ConverterFunc converter) { converters_.push_back(std::move(converter)); }

  const std::vector<ConverterFunc>& get_converters() const { return converters_; }

 protected:
  ConverterRegistryBase() = default;

 private:
  friend ConverterRegistryBase<From, To>& converterRegistryInstance<From, To>();

  std::vector<ConverterFunc> converters_;
};

// template <typename From, typename To>
// HAMI_EXPORT std::optional<To> convert(const From& from) {
//   std::optional<To> result;
//   for (const auto& converter : converterRegistryInstance<From, To>().get_converters()) {
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
    if (result) return result;
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

}  // namespace hami

// Register Python to C++ converter
#define HAMI_ADD_PY2CPP(converter_func)                                                          \
  static hami::ConverterRegister<pybind11::handle, hami::any> _hami_add_py2cpp_static##__LINE__( \
      converter_func)

// Register C++ to Python converter
#define HAMI_ADD_CPP2PY(converter_func)                                                          \
  static hami::ConverterRegister<hami::any, pybind11::object> _hami_add_cpp2py_static##__LINE__( \
      converter_func)

// Get all Python to C++ converters
#define HAMI_GET_PY2CPP() \
  hami::converterRegistryInstance<pybind11::handle, hami::any>().get_converters()

// Get all C++ to Python converters
#define HAMI_GET_CPP2PY() \
  hami::converterRegistryInstance<hami::any, pybind11::object>().get_converters()