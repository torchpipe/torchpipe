// Copyright 2021-2023 NetEase.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "base_logging.hpp"
#include "tensor_type_caster.hpp"
#include "ipipe_utils.hpp"
#include "any2object.hpp"
#include "object2any.hpp"

#ifdef PYBIND
#include <pybind11/pybind11.h>
#include <pybind11/cast.h>
#endif

// #include "Backend.hpp"

namespace ipipe {

void cast_type_v2(const any& data, py::dict result_dict, std::string key) {
  try {
    result_dict[py::str(key)] = any2object(data);
  } catch (...) {
    std::throw_with_nested(std::runtime_error("failed while return to python. key=" + key +
                                              ", c++ type=" + local_demangle(data.type().name())));
  }
  return;
}

void dict2py(dict input, pybind11::dict result_dict, bool keep_data) {
  if (result_dict.contains(TASK_RESULT_KEY)) {
    PyDict_DelItemString(result_dict.ptr(), TASK_RESULT_KEY);
  }

  for (auto iter = input->begin(); iter != input->end(); ++iter) {
    if (!keep_data && iter->first == TASK_DATA_KEY) {
      continue;
    }
    cast_type_v2(iter->second, result_dict, iter->first);
  }

  return;
}

dict py2dict(pybind11::dict input) {
  dict result_dict(new std::unordered_map<std::string, any>());

  for (auto& item : input) {
    std::string key = py::cast<std::string>(item.first);
    try {
      (*result_dict)[key] = object2any(item.second);
    } catch (...) {
      std::string err_msg =
          "During the process of converting the input[Python dict object] to a C++ object, "
          "unsupported type was discovered."
          " key: " +
          key + ", value: " + std::string(py::str(py::type::of(item.second)));
      // key + ", value: " + std::string(py::str(py::type::of(item.second)));

      std::throw_with_nested(std::runtime_error(err_msg));
    }
  }
  return result_dict;
}

}  // namespace ipipe

#ifdef PYBIND

namespace pybind11 {
namespace detail {
template <>
struct type_caster<ipipe::any> {
 public:
  PYBIND11_TYPE_CASTER(ipipe::any, _("ipipe::any"));

  // Python -> C++
  bool load(handle src, bool) {
    value = ipipe::object2any(src);
    return true;
  }

  // C++ -> Python
  static handle cast(const ipipe::any& src, return_value_policy /* policy */, handle /* parent */) {
    return ipipe::any2object(src).release();
  }
};

template <>
struct type_caster<ipipe::dict> {
 public:
  PYBIND11_TYPE_CASTER(ipipe::dict, _("ipipe::dict"));

  // Python -> C++
  bool load(handle src, bool) {
    auto dict = src.cast<py::dict>();
    auto map = std::make_shared<std::unordered_map<std::string, ipipe::any>>();
    for (auto item : dict) {
      (*map)[item.first.cast<std::string>()] = ipipe::object2any(item.second);
    }
    value = map;
    return true;
  }

  // C++ -> Python
  static handle cast(const ipipe::dict& src, return_value_policy /* policy */,
                     handle /* parent */) {
    py::dict dict;
    for (const auto& item : *src) {
      dict[item.first.c_str()] = ipipe::any2object(item.second);
    }
    return dict.release();
  }
};

}  // namespace detail
}  // namespace pybind11
#endif