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

#include "DynamicResizeMat.hpp"

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

bool DynamicResizeMat::init(const std::unordered_map<std::string, std::string>& config_param,
                            dict dict_config) {
  params_ = std::unique_ptr<Params>(new Params({}, {}, {}, {}));
  if (!params_->init(config_param)) return false;

  return true;
}

void DynamicResizeMat::forward(dict input_dict) {
  params_->check_and_update(input_dict);

  int resize_h;
  int resize_w;
  TRACE_EXCEPTION(resize_h = any_cast<int>(input_dict->at("resize_h")));
  TRACE_EXCEPTION(resize_w = any_cast<int>(input_dict->at("resize_w")));

  // auto resize_h = std::stoi(params_->operator[]("resize_h"));
  // auto resize_w = std::stoi(params_->operator[]("resize_w"));
  if (resize_h > 1024 * 1024 || resize_w > 1024 * 1024 || resize_h < 1 || resize_w < 1) {
    SPDLOG_ERROR("DynamicResizeMat: illigle h or w: h=" + std::to_string(resize_h) +
                 "w=" + std::to_string(resize_w));
    return;
  }

  auto& input = *input_dict;
  if (input[TASK_DATA_KEY].type() != typeid(cv::Mat)) {
    SPDLOG_ERROR("DynamicResizeMat: error input type: " +
                 std::string(input[TASK_DATA_KEY].type().name()));
    return;
  }
  auto data = any_cast<cv::Mat>(input[TASK_DATA_KEY]);

  cv::Mat im_resize;
  cv::resize(data, im_resize, cv::Size(resize_w, resize_h));

  input[TASK_RESULT_KEY] = im_resize;
  return;
}
IPIPE_REGISTER(Backend, DynamicResizeMat, "DynamicResizeMat");

}  // namespace ipipe