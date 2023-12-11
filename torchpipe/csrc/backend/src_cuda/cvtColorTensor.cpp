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

#include "cvtColorTensor.hpp"

#include <ATen/ATen.h>

#include <numeric>

#include "base_logging.hpp"
#include "reflect.h"
#include "torch_utils.hpp"
#include "exception.hpp"
namespace ipipe {

bool cvtColorTensor::init(const std::unordered_map<std::string, std::string>& config_param,
                          dict dict_config) {
  params_ = std::unique_ptr<Params>(new Params({}, {"color"}, {}, {}));
  if (!params_->init(config_param)) return false;
  color_ = params_->at("color");
  if (color_ != "rgb" && color_ != "bgr") {
    SPDLOG_ERROR("color must be rgb or bgr: " + color_);
    return false;
  }
  //   index_selecter_cpu_ = at::tensor({2, 1, 0}).toType(at::kLong);
  //   index_selecter_ = at::tensor({2, 1, 0}).toType(at::kLong).cuda();

  return true;
}

void cvtColorTensor::forward(dict input_dict) {
  std::string input_color;
  TRACE_EXCEPTION(input_color = any_cast<std::string>(input_dict->at("color")));
  if (input_color != "rgb" && input_color != "bgr") {
    throw std::invalid_argument("input_color should be rgb or bgr, but is " + input_color);
  }

  auto input_tensor = dict_get<at::Tensor>(input_dict, TASK_DATA_KEY);
  if (input_tensor.scalar_type() != at::kFloat) {
    input_tensor = input_tensor.to(at::kFloat);
  }
  // if (!input_tensor.is_contiguous()) {
  //   input_tensor = input_tensor.contiguous();
  // }

  if (is_hwc(input_tensor)) {
    if (color_ != input_color) {
      input_tensor = at::flip(input_tensor, {2});
    }
  } else if (is_nchw(input_tensor)) {
    if (color_ != input_color) {
      input_tensor = at::flip(input_tensor, {1});
    }
  } else {
    throw std::invalid_argument("input tensor should be hwc or nchw");
  }
  (*input_dict)["color"] = color_;
  (*input_dict)[TASK_RESULT_KEY] = input_tensor;
}
IPIPE_REGISTER(Backend, cvtColorTensor, "cvtColorTensor,CvtColorTensor");

}  // namespace ipipe