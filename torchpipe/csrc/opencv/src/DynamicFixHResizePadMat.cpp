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

#include "DynamicFixHResizePadMat.hpp"

#include <memory>
#include <vector>
#include "base_logging.hpp"
#include "params.hpp"

#include <fstream>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"

#define PALIGN_UP(x, align) ((x + (align - 1)) & ~(align - 1))

#include "reflect.h"
namespace ipipe {

bool DynamicFixHResizePadMat::init(const std::unordered_map<std::string, std::string>& config_param,
                                   dict dict_config) {
  params_ =
      std::unique_ptr<Params>(new Params({{"pad_value", "0"}}, {}, {}, {"resize_h", "max_w"}));
  if (!params_->init(config_param)) return false;

  auto pads = str_split(params_->at("pad_value"));
  for (const auto& item : pads) pad_values_.push_back(std::stof(item));

  if (pad_values_.empty()) return false;
  while (pad_values_.size() < 3) {
    pad_values_.push_back(pad_values_.back());
  }

  return true;
}

void DynamicFixHResizePadMat::forward(const std::vector<dict>& input_dicts) {
  for (auto input_dict : input_dicts) {
    params_->check_and_update(input_dict);
    auto& input = *input_dict;
    if (input[TASK_DATA_KEY].type() != typeid(cv::Mat)) {
      SPDLOG_ERROR("DynamicFixHResizePadMat: error input type: " +
                   std::string(input[TASK_DATA_KEY].type().name()));
      return;
    }
    auto resize_h = std::stoi(params_->at("resize_h"));
    auto max_w = std::stoi(params_->at("max_w"));

    if (max_w <= 0) {
      SPDLOG_ERROR("error:   max_w = {} ", max_w);
      return;
    }
    if (resize_h <= 0) {
      SPDLOG_ERROR("error:   resize_h = {} ", resize_h);
      return;
    }

    auto data = any_cast<cv::Mat>(input[TASK_DATA_KEY]);

    float ratio = float(data.cols) / float(data.rows);
    int resize_w;
    if (ceilf(resize_h * ratio) > max_w)
      resize_w = max_w;
    else
      resize_w = int(ceilf(resize_h * ratio));
    cv::Mat resize_img;
    cv::resize(data, resize_img, cv::Size(resize_w, resize_h), 0.f, 0.f, cv::INTER_LINEAR);
    if (resize_w < max_w) {
      cv::copyMakeBorder(resize_img, resize_img, 0, 0, 0, max_w - resize_w, cv::BORDER_CONSTANT,
                         cv::Scalar(pad_values_[0], pad_values_[1], pad_values_[2]));
    }

    input[TASK_RESULT_KEY] = resize_img;
  }
}

IPIPE_REGISTER(Backend, DynamicFixHResizePadMat, "DynamicFixHResizePadMat");

}  // namespace ipipe