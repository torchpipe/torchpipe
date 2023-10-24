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

#include "FloatMat.hpp"

#include <memory>
#include <vector>
#include "base_logging.hpp"
#include "params.hpp"

#include <fstream>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"

#include "reflect.h"
namespace ipipe {

void FloatMat::forward(dict input_dict) {
  auto& input = *input_dict;
  if (input[TASK_DATA_KEY].type() != typeid(cv::Mat)) {
    SPDLOG_ERROR("FloatMat: error input type: " + std::string(input[TASK_DATA_KEY].type().name()));
    return;
  }
  auto data = any_cast<cv::Mat>(input[TASK_DATA_KEY]);
  cv::Mat dst;
  data.convertTo(dst, CV_32F);

  input[TASK_RESULT_KEY] = dst;
}
IPIPE_REGISTER(Backend, FloatMat, "FloatMat");
}  // namespace ipipe