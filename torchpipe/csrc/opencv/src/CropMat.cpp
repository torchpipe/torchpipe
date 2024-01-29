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

#include "CropMat.hpp"

#include <vector>

#include "Backend.hpp"
#include "dict.hpp"

#include "base_logging.hpp"
#include "reflect.h"

#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"

namespace ipipe {

cv::Mat opencv_crop(cv::Mat in, int x1, int y1, int x2, int y2) {
  if (x2 <= x1 || y2 <= y1 || x2 > in.cols || y2 > in.rows) {
    throw std::invalid_argument("Invalid coordinates for cropping.");
  }

  cv::Rect roi(x1, y1, x2 - x1, y2 - y1);

  cv::Mat out = in(roi);

  return out;
}

void CropMat::forward(dict input_dict) {
  auto& input = *input_dict;

  std::vector<int> pbox = dict_get<std::vector<int>>(input_dict, TASK_BOX_KEY);

  auto input_mat = dict_get<cv::Mat>(input_dict, TASK_DATA_KEY);

  if (pbox.size() < 4) {
    SPDLOG_ERROR("TASK_BOX_KEY: boxes[i].size() < 4");
    throw std::invalid_argument("get an error box");
  }

  const int& x1 = pbox[0];
  const int& y1 = pbox[1];
  const int& x2 = pbox[2];
  const int& y2 = pbox[3];
  auto cropped = opencv_crop(input_mat, x1, y1, x2, y2);
  if (cropped.total() <= 0) {
    SPDLOG_ERROR("get an empty mat");
    throw std::runtime_error("get an empty mat");
  }

  input[TASK_RESULT_KEY] = cropped;
}

IPIPE_REGISTER(Backend, CropMat, "CropMat");
}  // namespace ipipe
