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

#include "PillowResizeMat.hpp"
#include "pillow-resize/PillowResize.hpp"

#include <memory>
#include <vector>
#include "base_logging.hpp"
#include "params.hpp"
#include "exception.hpp"

#include <fstream>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "reflect.h"
namespace ipipe {

bool PillowResizeMat::init(const std::unordered_map<std::string, std::string>& config_param,
                           dict dict_config) {
  params_ = std::unique_ptr<Params>(new Params({}, {"resize_h", "resize_w"}, {}, {}));
  if (!params_->init(config_param)) return false;

  TRACE_EXCEPTION(resize_h_ = std::stoi(params_->operator[]("resize_h")));
  TRACE_EXCEPTION(resize_w_ = std::stoi(params_->operator[]("resize_w")));
  if (resize_h_ > 1024 * 1024 || resize_w_ > 1024 * 1024 || resize_h_ < 1 || resize_w_ < 1) {
    SPDLOG_ERROR("PillowResizeMat: illigle h or w: h=" + std::to_string(resize_h_) +
                 "w=" + std::to_string(resize_w_));
    return false;
  }

  return true;
}

void PillowResizeMat::forward(dict input_dict) {
  params_->check_and_update(input_dict);
  auto& input = *input_dict;
  if (input[TASK_DATA_KEY].type() != typeid(cv::Mat)) {
    SPDLOG_ERROR("PillowResizeMat: error input type: " +
                 std::string(input[TASK_DATA_KEY].type().name()));
    return;
  }
  auto data = any_cast<cv::Mat>(input[TASK_DATA_KEY]);

  IPIPE_ASSERT(data.type() == CV_8UC3 && !data.empty());
  cv::Mat im_resize;
  if (!data.isContinuous()) {
    data = data.clone();
  }

  SPDLOG_DEBUG("data.cols = {} data.rows = {} ", data.cols, data.rows);

  // cv::resize(data, im_resize, cv::Size(resize_w_, resize_h_));
  im_resize = PillowResize::resize(data, cv::Size(resize_w_, resize_h_),
                                   PillowResize::InterpolationMethods::INTERPOLATION_BILINEAR);

  IPIPE_ASSERT(!im_resize.empty());
  input[TASK_RESULT_KEY] = im_resize;
}
IPIPE_REGISTER(Backend, PillowResizeMat, "PillowResizeMat");
}  // namespace ipipe