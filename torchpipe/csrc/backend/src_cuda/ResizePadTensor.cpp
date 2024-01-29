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

#include "ResizePad.hpp"

#include <memory>
#include <vector>
#include "base_logging.hpp"
#include "params.hpp"

#include <fstream>
#include <ATen/ATen.h>

#include "reflect.h"
#include "base_logging.hpp"
#include "torch_utils.hpp"

namespace ipipe {
/**
 * @brief 参见 @ref ResizePad. 接收和输出torch.Tensor
 *
 */
class ResizePadTensor : public ResizePad {
  void forward(dict input_dict) {
    params_->check_and_update(input_dict);

    auto input_tensor = dict_get<at::Tensor>(input_dict, TASK_DATA_KEY);

    input_tensor = img_1chw_guard(input_tensor).to(at::kFloat);
    if (!input_tensor.is_contiguous()) input_tensor = input_tensor.contiguous();

    int cols = input_tensor.size(-1);
    int rows = input_tensor.size(-2);

    float ratio = float(cols) / float(rows);
    int resize_h;
    int resize_w;
    if (max_h_ * ratio <= max_w_) {
      resize_w = std::min(int(std::round(max_h_ * ratio)), max_w_);
      resize_h = max_h_;
    } else {
      resize_w = max_w_;
      resize_h = std::min(int(std::round(max_w_ / ratio)), max_h_);
    }
    at::Tensor resize_img;
    float x_ratio = cols * 1.0f / resize_w;
    float y_ratio = rows * 1.0f / resize_h;
    std::function<std::pair<float, float>(float, float)> inverse_trans = [x_ratio, y_ratio](
                                                                             float x, float y) {
      return std::pair<float, float>(x_ratio * x, y_ratio * y);
    };
    (*input_dict)["inverse_trans"] = inverse_trans;

    if (resize_w == cols && resize_h == rows) {
      resize_img = input_tensor;
    } else {
      resize_img = at::upsample_bilinear2d(input_tensor, {resize_h, resize_w}, true);
    }
    // https://github.com/openppl-public/ppl.cv/blob/master/src/ppl/cv/cuda/copymakeborder.cu

    if ((resize_w < max_w_ || resize_h < max_h_) && !pad_values_.empty()) {
      resize_img = at::constant_pad_nd(resize_img, {0, max_w_ - resize_w, 0, max_h_ - resize_h},
                                       pad_values_[0]);
    }

    (*input_dict)[TASK_RESULT_KEY] = resize_img;
  }
};

IPIPE_REGISTER(Backend, ResizePadTensor, "ResizePadTensor");

}  // namespace ipipe