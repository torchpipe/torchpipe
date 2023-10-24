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

// #define PYBIND
// from
// https://stackoverflow.com/questions/60949451/how-to-send-a-cvmat-to-python-over-shared-memory
// https://github.com/pthom/cvnp/blob/fb6ece745c7cfc1be8e5743fe5cbd55db08805b0/cvnp/cvnp.cpp
#ifdef PYBIND
#include "mat2numpy.hpp"

namespace ipipe {

#ifdef WITH_OPENCV

py::dtype determine_np_dtype(int depth) {
  switch (depth) {
    case CV_8U:
      return py::dtype::of<uint8_t>();
    case CV_8S:
      return py::dtype::of<int8_t>();
    case CV_16U:
      return py::dtype::of<uint16_t>();
    case CV_16S:
      return py::dtype::of<int16_t>();
    case CV_32S:
      return py::dtype::of<int32_t>();
    case CV_32F:
      return py::dtype::of<float>();
    case CV_64F:
      return py::dtype::of<double>();
    default:
      throw std::invalid_argument("Unsupported data type.");
  }
}

std::vector<std::size_t> determine_shape(const cv::Mat& m) {
  //   if (m.channels() == 1) {
  //     return {static_cast<size_t>(m.rows), static_cast<size_t>(m.cols)};
  //   }

  return {static_cast<size_t>(m.rows), static_cast<size_t>(m.cols),
          static_cast<size_t>(m.channels())};
}

py::capsule make_capsule_mat(const cv::Mat& m) {
  return py::capsule(new cv::Mat(m), [](void* v) { delete reinterpret_cast<cv::Mat*>(v); });
}

py::array mat2numpy(const cv::Mat& m) {
  if (!m.isContinuous() && !m.empty()) {
    auto z = m.clone();
    return py::array(determine_np_dtype(z.depth()), determine_shape(z), z.data,
                     make_capsule_mat(m));

    // throw std::invalid_argument("Only continuous Mats supported.");
  }

  return py::array(determine_np_dtype(m.depth()), determine_shape(m), m.data, make_capsule_mat(m));
}

// py::array mat2numpy_deep(const cv::Mat& m) {
//   da.data, {da.rows, da.cols, da.channels()}, da.elemSize1() == 1 ? at::kByte : at::kFloat;
//   py::array_t<T, py::array::c_style | py::array::forcecast> array(da.rows * da.cols *
//                                                                   da.channels() *
//                                                                   da.elemSize1());
//   py::buffer_info buf3 = array.request();
//   T* dst = static_cast<T*>(buf3.ptr);
//   std::memcpy(dst, data.data(), data.size() * sizeof(T));

//   return array;

//   if (!m.isContinuous() && !m.empty()) {
//     auto z = m.clone();
//     return py::array(determine_np_dtype(z.depth()), determine_shape(z), z.data);

//     // throw std::invalid_argument("Only continuous Mats supported.");
//   }

//   return py::array(determine_np_dtype(m.depth()), determine_shape(m), m.data);
// }
#endif

}  // namespace ipipe

#endif