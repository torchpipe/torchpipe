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

#include "PerspectiveTransformMat.hpp"
#include "Backend.hpp"
#include "base_logging.hpp"
#include "ipipe_utils.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"

namespace ipipe {

class PerspectiveTransformMat : public SingleBackend {
 public:
  void forward(dict input_dict) override {
    auto& input = *input_dict;
    if (input[TASK_DATA_KEY].type() != typeid(cv::Mat)) {
      SPDLOG_ERROR("need cv::Mat, get: " +
                   ipipe::local_demangle(input[TASK_DATA_KEY].type().name()));
      return;
    }
    auto iter = input_dict->find(TASK_BOX_KEY);
    if (iter == input_dict->end()) {
      SPDLOG_ERROR("PerspectiveTransformMat: TASK_BOX_KEY not found.");
      return;
    }

    std::vector<std::vector<int>> points = any_cast<std::vector<std::vector<int>>>(iter->second);
    IPIPE_ASSERT(points.size() == 4 && points[0].size() == 2 && points[1].size() == 2 &&
                 points[2].size() == 2 && points[3].size() == 2);
    cv::Mat img = any_cast<cv::Mat>(input[TASK_DATA_KEY]);
    std::vector<cv::Mat> cropped_imgs;

    int img_crop_width =
        int(sqrt(pow(points[0][0] - points[1][0], 2) + pow(points[0][1] - points[1][1], 2)));
    int img_crop_height =
        int(sqrt(pow(points[0][0] - points[3][0], 2) + pow(points[0][1] - points[3][1], 2)));

    cv::Point2f pts_std[4];
    pts_std[0] = cv::Point2f(0., 0.);
    pts_std[1] = cv::Point2f(img_crop_width, 0.);
    pts_std[2] = cv::Point2f(img_crop_width, img_crop_height);
    pts_std[3] = cv::Point2f(0.f, img_crop_height);

    cv::Point2f pointsf[4];
    pointsf[0] = cv::Point2f(points[0][0], points[0][1]);
    pointsf[1] = cv::Point2f(points[1][0], points[1][1]);
    pointsf[2] = cv::Point2f(points[2][0], points[2][1]);
    pointsf[3] = cv::Point2f(points[3][0], points[3][1]);

    cv::Mat M = cv::getPerspectiveTransform(pointsf, pts_std);

    cv::Mat dst_img;
    cv::warpPerspective(img, dst_img, M, cv::Size(img_crop_width, img_crop_height),
                        cv::BORDER_CONSTANT, 0);
    // cropped_imgs.push_back(dst_img);

    input[TASK_RESULT_KEY] = dst_img;
  }
};

IPIPE_REGISTER(Backend, PerspectiveTransformMat, "PerspectiveTransformMat");

}  // namespace ipipe