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

#include "ResizePad.hpp"

#include <memory>
#include <vector>
#include "base_logging.hpp"
#include "params.hpp"

#include <fstream>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"

#include "reflect.h"
namespace ipipe {
/**
 * @brief 保持长宽比resize并进行中心对齐的pad. 参见 @ref ResizePad. 接收和输出
 cv::Mat.

 *
 */
class ResizeCenterPadMat : public ResizePad {
  void forward(dict input_dict) {
    params_->check_and_update(input_dict);
    auto& input = *input_dict;
    if (input[TASK_DATA_KEY].type() != typeid(cv::Mat)) {
      SPDLOG_ERROR("ResizeCenterPadMat: error input type: " +
                   std::string(input[TASK_DATA_KEY].type().name()));
      return;
    }
    auto data = any_cast<cv::Mat>(input[TASK_DATA_KEY]);

    float ratio = float(data.cols) / float(data.rows);
    int resize_h;
    int resize_w;
    if (max_h_ * ratio <= max_w_) {
      resize_w = std::min(int(std::round(max_h_ * ratio)), max_w_);
      resize_h = max_h_;
    } else {
      resize_w = max_w_;
      resize_h = std::min(int(std::round(max_w_ / ratio)), max_h_);
    }
    cv::Mat resize_img;
    float x_ratio = data.cols * 1.0f / resize_w;
    float y_ratio = data.rows * 1.0f / resize_h;
    int top = (max_h_ - resize_h) / 2;
    int left = (max_w_ - resize_w) / 2;
    const int h = data.rows;
    const int w = data.cols;
    std::function<std::pair<float, float>(float, float)> inverse_trans =
        [x_ratio, y_ratio, top, left, h, w](float x, float y) {
          float a = x_ratio * (x - left);
          float b = y_ratio * (y - top);
          if (a < 0) a = 0;
          if (b < 0) b = 0;
          if (a >= w) a = w - 1;
          if (b >= h) b = h - 1;
          return std::pair<float, float>(a, b);
        };
    input["inverse_trans"] = inverse_trans;
    if (resize_w == data.cols && resize_h == data.rows) {
      resize_img = data;
    } else
      cv::resize(data, resize_img, cv::Size(resize_w, resize_h), 0.f, 0.f, cv::INTER_LINEAR);
    if ((resize_w < max_w_ || resize_h < max_h_) && !pad_values_.empty()) {
      if (pad_values_[0] - int(pad_values_[0]) > 1e-5 ||
          pad_values_[1] - int(pad_values_[1]) > 1e-5 ||
          pad_values_[2] - int(pad_values_[2]) > 1e-5) {
        cv::Mat img2;
        resize_img.convertTo(img2, CV_32FC3);  // or CV_32F works (too)
        cv::copyMakeBorder(img2, resize_img, top, max_h_ - resize_h - top, left,
                           max_w_ - resize_w - left, cv::BORDER_CONSTANT,
                           cv::Scalar(pad_values_[0], pad_values_[1], pad_values_[2]));
      } else {
        cv::copyMakeBorder(resize_img, resize_img, top, max_h_ - resize_h - top, left,
                           max_w_ - resize_w - left, cv::BORDER_CONSTANT,
                           cv::Scalar(pad_values_[0], pad_values_[1], pad_values_[2]));
      }
    }

    input[TASK_RESULT_KEY] = resize_img;
  }
};

IPIPE_REGISTER(Backend, ResizeCenterPadMat, "ResizeCenterPadMat");

}  // namespace ipipe