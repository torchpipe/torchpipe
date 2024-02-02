// Copyright 2021-2024 NetEase.
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

#pragma once

#ifdef WITH_OPENCV
#include "opencv2/core.hpp"
#endif
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

// pybind11/pybind11.h
namespace py = pybind11;
namespace ipipe {



template <class T, typename std::enable_if<((std::is_arithmetic<T>::value)), int>::type = 0>
py::array_t<T> list2numpy(const std::vector<T>& data) {
  py::array_t<T, py::array::c_style | py::array::forcecast> array(data.size());
  py::buffer_info buf3 = array.request();
  T* dst = static_cast<T*>(buf3.ptr);
  std::memcpy(dst, data.data(), data.size() * sizeof(T));

  return array;
}

template <class T, typename std::enable_if<((std::is_arithmetic<T>::value)), int>::type = 0>
py::list list22numpy(const std::vector<std::vector<T>>& data) {
  py::list result;
  for (const auto& item : data) {
    result.append(list2numpy(item));
  }
  return result;

  // std::size_t h = data.size();
  // std::size_t w = h == 0 ? 0 : data[0].size();

  // py::array_t<T, py::array::c_style | py::array::forcecast> array(h * w);
  // py::buffer_info buf3 = array.request();
  // T* dst = static_cast<T*>(buf3.ptr);
  // for (const auto& item : data) {
  //   if (item.size() != w) {
  //     throw py::type_error("list22numpy: not aligned.");
  //   }
  //   std::memcpy(dst, item.data(), w * sizeof(T));
  //   dst += w;
  // }
  // array = array.reshape({h, w});

  // return array;
}

template <class T, typename std::enable_if<((std::is_arithmetic<T>::value)), int>::type = 0>
py::list list32numpy(const std::vector<std::vector<std::vector<T>>>& data) {
  py::list results;
  for (const auto& item : data) {
    py::list result;
    for (const auto& item_inner : item) {
      result.append(list2numpy(item_inner));
    }
    results.append(result);
  }
  return results;
}

}  // namespace ipipe