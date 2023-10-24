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

#pragma once

#include "dict.hpp"

#include "pybind11/pybind11.h"
namespace py = pybind11;

// #ifdef PYBIND
// #include <opencv2/core/core.hpp>
// namespace ipipe {
// struct Mat {
//   Mat() = default;
//   Mat(cv::Mat data) : mat(data) {}
//   cv::Mat mat;
//   operator cv::Mat() const { return mat; }
//   std::pair<int, int> size() { return std::pair<int, int>(mat.cols, mat.rows); }
//   int channels() { return mat.channels(); }
// };
// }  // namespace ipipe
// #endif
namespace ipipe {

// https://github.com/pytorch/pytorch/blob/30fb2c4abaaaa966999eab11674f25b18460e609/torch/csrc/cuda/python_nccl.cpp
// static inline at::Tensor extract_tensor(PyObject* obj) {
//   if (!THPVariable_Check(obj)) {
//     throw torch::TypeError("expected Tensor (got %s)", Py_TYPE(obj)->tp_name);
//   }
//   return THPVariable_Unpack(obj);
// }

// static inline std::vector<at::Tensor> extract_tensors(PyObject* obj) {
//   auto seq = THPObjectPtr(PySequence_Fast(obj, "expected a sequence"));
//   if (!seq) throw python_error();

//   std::vector<at::Tensor> list;
//   Py_ssize_t length = PySequence_Fast_GET_SIZE(seq.get());
//   for (Py_ssize_t i = 0; i < length; i++) {
//     PyObject* item = PySequence_Fast_GET_ITEM(seq.get(), i);
//     if (!THPVariable_Check(item)) {
//       throw torch::TypeError("expected Tensor at %d (got %s)", (int)i, Py_TYPE(item)->tp_name);
//     }
//     list.emplace_back(THPVariable_Unpack(item));
//   }
//   return list;
// }

dict py2dict(pybind11::dict input);

void dict2py(dict input, pybind11::dict result_dict);
}  // namespace ipipe
