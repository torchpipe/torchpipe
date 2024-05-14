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

#include "PillowResizeTensor.hpp"

#include <memory>
#include <vector>
#include "base_logging.hpp"
#include "params.hpp"
#include "torch_utils.hpp"
#include "exception.hpp"

#include <fstream>
#include <memory>
#include "ipipe_common.hpp"

#include "reflect.h"
namespace ipipe {

bool PillowResizeTensor::init(const std::unordered_map<std::string, std::string>& config_param,
                              dict dict_config) {
  params_ = std::unique_ptr<Params>(new Params({}, {"resize_h", "resize_w"}, {}, {}));
  if (!params_->init(config_param)) return false;

  TRACE_EXCEPTION(resize_h_ = std::stoi(params_->operator[]("resize_h")));
  TRACE_EXCEPTION(resize_w_ = std::stoi(params_->operator[]("resize_w")));
  if (resize_h_ > 1024 * 1024 || resize_w_ > 1024 * 1024 || resize_h_ < 1 || resize_w_ < 1) {
    SPDLOG_ERROR("PillowResizeTensor: illigle h or w: h=" + std::to_string(resize_h_) +
                 "w=" + std::to_string(resize_w_));
    return false;
  }

  resizer_ = std::make_unique<ipipe_nvcv::PillowResizeCudaV2>(
      torch::IntArrayRef({1, 3, 5000, 5000}), resize_h_, resize_w_, torch::kFloat);
  if (!resizer_) return false;
  return true;
}

void PillowResizeTensor::forward(dict input_dict) {
  params_->check_and_update(input_dict);
  auto& input = *input_dict;
  if (input[TASK_DATA_KEY].type() != typeid(torch::Tensor)) {
    SPDLOG_ERROR("PillowResizeTensor: error input type: " +
                 std::string(input[TASK_DATA_KEY].type().name()));
    return;
  }
  auto data = any_cast<torch::Tensor>(input[TASK_DATA_KEY]);
  IPIPE_ASSERT(data.scalar_type() == torch::kByte);

  auto im_resize = img_1hwc_guard(data);
  if (!im_resize.is_cuda()) im_resize.cuda();
  // if (im_resize.scalar_type() != torch::kFloat) {
  //   im_resize = im_resize.to(torch::kFloat, true, false, torch::MemoryFormat::Contiguous);
  // } else {
  //   im_resize = im_resize.contiguous();
  // }
  if (im_resize.size(1) != resize_h_ || im_resize.size(2) != resize_w_) {
    im_resize = im_resize.contiguous();

    // cv::resize(data, im_resize, cv::Size(resize_w_, resize_h_));
    im_resize = resizer_->forward(im_resize);

    if (im_resize.sizes().size() != 4 ||
        im_resize.size(0) * im_resize.size(1) * im_resize.size(2) * im_resize.size(3) <= 0) {
      SPDLOG_ERROR("im_resize.cols={}, im_resize.rows={}, im_resize.channels={}", im_resize.size(2),
                   im_resize.size(1), im_resize.size(3));
      return;
    }
  }
  im_resize = im_resize.permute({0, 3, 1, 2});
  input[TASK_RESULT_KEY] = im_resize;
}
IPIPE_REGISTER(Backend, PillowResizeTensor, "PillowResizeTensor");
}  // namespace ipipe