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

#include "CropTensor.hpp"

#include <vector>

#include "c10/cuda/CUDAStream.h"
// #include <ATen/cuda/CUDAContext.h>

#include "cuda_runtime.h"

#include <ATen/ATen.h>

#include "Backend.hpp"
#include "dict.hpp"

#include "base_logging.hpp"
#include "reflect.h"
#include "torch_utils.hpp"
namespace ipipe {

at::Tensor tensor_crop(at::Tensor input, int x1, int y1, int x2, int y2) {
  const int roi_w = x2 - x1;
  const int roi_h = y2 - y1;

  const auto img_w = input.size(1);

  auto options = at::TensorOptions()
                     .device(at::kCUDA, -1)
                     .dtype(input.dtype())  // scalar_type dtype at::kByte
                     .layout(at::kStrided)
                     .requires_grad(false);
  at::Tensor image_tensor = at::empty({roi_h, roi_w, 3},  //, max.d[2], max.d[3]
                                      options, at::MemoryFormat::Contiguous);
  if (input.dtype() == at::kByte) {
    const unsigned char* p_src = input.data_ptr<unsigned char>();
    unsigned char* p_dst = image_tensor.data_ptr<unsigned char>();

    if (!input.is_contiguous()) {
      input = input.contiguous();
    }

    cudaMemcpy2DAsync(p_dst,                            // 目的指针
                      roi_w * 3,                        // 目的pitch
                      p_src + x1 * 3 + y1 * 3 * img_w,  // 源指针
                      img_w * 3,                        // 源数据pitch
                      roi_w * 3,                        // 数据拷贝宽度
                      roi_h,                            // 数据拷贝高度
                      cudaMemcpyDeviceToDevice,
                      c10::cuda::getCurrentCUDAStream());  // 从CPU拷贝二维数组到GPU上

  } else {
    float* p_src = input.data_ptr<float>();
    float* p_dst = image_tensor.data_ptr<float>();

    if (!input.is_contiguous()) {
      input = input.contiguous();
    }

    cudaMemcpy2DAsync(p_dst,                            // 目的指针
                      sizeof(float) * roi_w * 3,        // 目的pitch
                      p_src + x1 * 3 + y1 * 3 * img_w,  // 源指针
                      sizeof(float) * img_w * 3,        // 源数据pitch
                      sizeof(float) * roi_w * 3,        // 数据拷贝宽度
                      roi_h,                            // 数据拷贝高度
                      cudaMemcpyDeviceToDevice,
                      c10::cuda::getCurrentCUDAStream());  // 从CPU拷贝二维数组到GPU上
  }

  return image_tensor;
}
// https://pytorch.org/cppdocs/notes/tensor_indexing.html
at::Tensor libtorch_crop(at::Tensor input, int x1, int y1, int x2, int y2) {
  if (input.sizes().size() >= 2) {  //..hw
    return input.index({"...", at::indexing::Slice(y1, y2), at::indexing::Slice(x1, x2)});
  } else {
    std::stringstream ss;
    ss << "input.sizes() = " << input.sizes() << " x1 y1 x2 y2 = " << x1 << " " << y1 << " " << x2
       << " " << y2;
    throw std::invalid_argument(ss.str());
  }
}

void CropTensor::forward(dict input_dict) {
  auto& input = *input_dict;

  std::vector<int> pbox = dict_get<std::vector<int>>(input_dict, TASK_BOX_KEY);

  auto input_tensor = dict_get<at::Tensor>(input_dict, TASK_DATA_KEY);

  input_tensor = img_1chw_guard(input_tensor);

  if (pbox.size() < 4) {
    SPDLOG_ERROR("TASK_BOX_KEY: boxes[i].size() < 4");
    throw std::invalid_argument("get an error box");
  }

  const uint32_t& x1 = pbox[0];
  const uint32_t& y1 = pbox[1];
  const uint32_t& x2 = pbox[2];
  const uint32_t& y2 = pbox[3];
  auto cropped = tensor_crop(input_tensor, x1, y1, x2, y2);
  if (cropped.numel() <= 0) {
    SPDLOG_ERROR("get an empty tensor");
    throw std::runtime_error("get an empty tensor");
  }

  input[TASK_RESULT_KEY] = cropped;
}

IPIPE_REGISTER(Backend, CropTensor, "CropTensor");
}  // namespace ipipe
