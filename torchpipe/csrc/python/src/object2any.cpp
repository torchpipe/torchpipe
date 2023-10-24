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

#include "object2any.hpp"
#include "ipipe_common.hpp"
#include "ipipe_utils.hpp"
#include "dict.hpp"
#include <torch/extension.h>
#include "base_logging.hpp"
#include <ATen/ATen.h>
#include <c10/util/Type.h>
#include <torch/csrc/autograd/python_variable.h>
#include "pybind11/numpy.h"
#include "pybind11/stl.h"
#include "event.hpp"

namespace py = pybind11;

namespace ipipe {
template <template <class... Args> class container>
any list_cast(py::handle second) {
  py::list list_data = py::cast<py::list>(second);
  int len = py::len(list_data);
  if (0 == len) {
    SPDLOG_DEBUG("not enough information to know the type inside this container.");
    return any(UnknownContainerTag());
  }

  PyObject* obj = (*list_data.begin()).ptr();

  if (THPVariable_Check(obj)) {
    return py::cast<container<at::Tensor>>(second);
  } else if (py::isinstance<py::str>((*list_data.begin()))) {
    return py::cast<container<std::string>>(second);
  } else if (py::isinstance<py::bytes>((*list_data.begin()))) {
    return py::cast<container<std::string>>(second);
  } else if (py::isinstance<py::bool_>((*list_data.begin()))) {
    return py::cast<container<bool>>(second);
  } else if (py::isinstance<py::int_>((*list_data.begin()))) {
    return py::cast<container<int>>(second);
  } else if (py::isinstance<py::float_>((*list_data.begin()))) {
    return py::cast<container<float>>(second);  // precision loss
  } else if (py::isinstance<py::array>(second)) {
    IPIPE_THROW(
        "array is not convertable to c++. You may use torch.from_numpy(...) to prepare data. ");
  }
  IPIPE_THROW("It is not able to convert " +
              std::string(py::str(py::type::of(*list_data.begin()))) + " from python to c++.");
}

template <class TT>
using LocalStrMap = std::unordered_map<std::string, TT>;

template <class T, template <class... Args> class container>
any LocalStrMap_cast(py::handle second) {
  T list_data = py::cast<T>(second);
  if (0 == py::len(list_data)) {
    SPDLOG_DEBUG("not enough information to know the type inside this container.");

    return any(UnknownContainerTag());
  }

  PyObject* obj = (list_data.begin()->second).ptr();

  if (THPVariable_Check(obj)) {
    return py::cast<container<at::Tensor>>(second);
  } else if (py::isinstance<py::str>(list_data.begin()->second)) {
    return py::cast<container<std::string>>(second);
  } else if (py::isinstance<py::bytes>(list_data.begin()->second)) {
    return py::cast<container<std::string>>(second);
  } else if (py::isinstance<py::bool_>(list_data.begin()->second)) {
    return py::cast<container<bool>>(second);
  } else if (py::isinstance<py::int_>(list_data.begin()->second)) {
    return py::cast<container<int>>(second);
  } else if (py::isinstance<py::float_>(list_data.begin()->second)) {
    return py::cast<container<float>>(second);  // precision
  } else if (py::isinstance<py::array>(second)) {
    IPIPE_THROW(
        "array is not convertable to c++. You may use torch.from_numpy(...) to prepare data. ");
  }
  IPIPE_THROW("It is not able to convert " + std::string(py::str(second)) + " from python to c++.");
}

any set_cast(py::handle second) {
  py::set list_data = py::cast<py::set>(second);

  if (0 == py::len(list_data)) {
    SPDLOG_DEBUG("not enough information to know the type inside this container");
    return any(UnknownContainerTag());
  }

  PyObject* obj = (*list_data.begin()).ptr();
  if (THPVariable_Check(obj)) {
    IPIPE_THROW("at::Tensor is unhashable.");
  } else if (py::isinstance<py::str>((*list_data.begin()))) {
    return py::cast<std::unordered_set<std::string>>(second);
  } else if (py::isinstance<py::bytes>((*list_data.begin()))) {
    return py::cast<std::unordered_set<std::string>>(second);
  } else if (py::isinstance<py::bool_>((*list_data.begin()))) {
    return py::cast<std::unordered_set<bool>>(second);
  } else if (py::isinstance<py::int_>((*list_data.begin()))) {
    return py::cast<std::unordered_set<int>>(second);
  } else if (py::isinstance<py::float_>((*list_data.begin()))) {
    return py::cast<std::unordered_set<float>>(second);  // percision loss
  } else if (py::isinstance<py::array>(second)) {
    IPIPE_THROW("array is not convertable to c++. ");
  }
  IPIPE_THROW("It is not able to convert " + std::string(py::str(second)) + " from python to c++.");
}
any object2any(pybind11::handle data) {
  if (py::isinstance<SimpleEvents>(data)) {
    return py::cast<std::shared_ptr<SimpleEvents>>(data);
  } else if (py::isinstance<py::list>(data)) {
    return list_cast<std::vector>(data);
  } else if (py::isinstance<py::set>(data)) {
    return set_cast(data);
  } else if (py::isinstance<py::dict>(data)) {
    return LocalStrMap_cast<py::dict, LocalStrMap>(data);
  } else if (py::isinstance<py::str>(data)) {
    return py::cast<std::string>(data);
  } else if (py::isinstance<py::bytes>(data)) {
    return py::cast<std::string>(data);
  } else if (THPVariable_Check(data.ptr())) {
    return py::cast<at::Tensor>(data);
  } else if (py::isinstance<py::bool_>(data)) {
    return py::cast<bool>(data);
  } else if (py::isinstance<py::int_>(data)) {
    return py::cast<int>(data);
  } else if (py::isinstance<py::float_>(data)) {
    return (float)py::cast<double>(data);  // change to float
  } else if (py::isinstance<ipipe::any>(data)) {
    return py::cast<ipipe::any>(data);
  } else if (py::isinstance<py::array>(data)) {
    IPIPE_THROW(
        "array is not convertable to c++. You may use torch.from_numpy(...) to prepare data. ");
  }
  IPIPE_THROW("It is not able to convert " + std::string(py::str(py::type::of(data))) +
              " from python to c++.");
}

}  // namespace ipipe

// // from https://github.com/pthom/cvnp/blob/master/cvnp/cvnp.cpp
// bool is_array_contiguous(const pybind11::array& a) {
//   pybind11::ssize_t expected_stride = a.itemsize();
//   for (int i = a.ndim() - 1; i >= 0; --i) {
//     pybind11::ssize_t current_stride = a.strides()[i];
//     if (current_stride != expected_stride) return false;
//     expected_stride = expected_stride * a.shape()[i];
//   }
//   return true;
// }