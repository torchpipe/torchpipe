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

#include "AlignMat.hpp"

#include <memory>
#include <vector>
#include "base_logging.hpp"
#include "params.hpp"
#include "exception.hpp"

#include <fstream>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "ipipe_utils.hpp"
// #include <c10/util/Type.h>

#include "reflect.h"
namespace ipipe {

bool AlignMat::init(const std::unordered_map<std::string, std::string>& config_param,
                    dict dict_config) {
  params_ = std::unique_ptr<Params>(new Params({{"align", "32"}}, {"resize_h", "max_w"}, {}, {}));
  if (!params_->init(config_param)) return false;
  TRACE_EXCEPTION(max_w_ = std::stoi(params_->at("max_w")));
  TRACE_EXCEPTION(resize_h_ = std::stoi(params_->at("resize_h")));
  TRACE_EXCEPTION(align_ = std::stoi(params_->at("align")));
  IPIPE_ASSERT(align_ > 0 && max_w_ > 0);
  if (max_w_ % align_ != 0) {
    SPDLOG_ERROR("max_w_%align_ != 0");
    return false;
  }
  return true;
}

void AlignMat::forward(const std::vector<dict>& input_dicts) {
  int target_w = 0;
  for (auto input_dict : input_dicts) {
    // params_->check_and_update(input_dict);
    auto& input = *input_dict;
    if (input[TASK_DATA_KEY].type() != typeid(cv::Mat)) {
      SPDLOG_ERROR("AlignMat: error input type: " +
                   std::string(input[TASK_DATA_KEY].type().name()));
      throw std::runtime_error("AlignMat: error input type: " +
                               std::string(input[TASK_DATA_KEY].type().name()));
    }
    cv::Mat data = any_cast<cv::Mat>(input[TASK_DATA_KEY]);

    target_w = std::max(target_w, int(ceilf(resize_h_ * data.cols) / float(data.rows)));
    if (target_w % align_ != 0) target_w = (target_w / align_ + 1) * align_;
    if (target_w > max_w_) {
      target_w = max_w_;
    }
  }
  for (auto input_dict : input_dicts) {
    auto& input = *input_dict;
    input[TASK_RESULT_KEY] = input[TASK_DATA_KEY];
    input["resize_w"] = target_w;
    input["resize_h"] = resize_h_;
  }
}
IPIPE_REGISTER(Backend, AlignMat, "AlignMat");

}  // namespace ipipe