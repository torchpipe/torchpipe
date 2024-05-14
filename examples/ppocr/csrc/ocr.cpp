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

#include "ATen/ATen.h"
// #include "base_logging.hpp"
#include <algorithm>
#include "Backend.hpp"
#include "dict.hpp"
#include "include/postprocess_op.h"
#include "params.hpp"
#include "prepost.hpp"
#include "reflect.h"
namespace ipipe {

class ResizeImgType0Mat : public SingleBackend {
 private:
  unsigned max_size_len_{0};

 public:
  bool init(const std::unordered_map<std::string, std::string>& config,
            dict /*dict_config*/) override {
    auto iter = config.find("max_size_len");
    if (iter == config.end()) return false;

    max_size_len_ = std::stoi(iter->second);

    return true;
  };
  void forward(dict input_dict) override {
    auto& input = *input_dict;
    if (input[TASK_DATA_KEY].type() != typeid(cv::Mat)) {
      return;
    }
    cv::Mat img = any_cast<cv::Mat>(input[TASK_DATA_KEY]);
    cv::Mat resize_img;

    int w = img.cols;
    int h = img.rows;

    float ratio = 1.f;
    int max_wh = w >= h ? w : h;
    if (max_wh > max_size_len_) {
      if (h > w) {
        ratio = float(max_size_len_) / float(h);
      } else {
        ratio = float(max_size_len_) / float(w);
      }
    }

    int resize_h = int(float(h) * ratio);
    int resize_w = int(float(w) * ratio);

    resize_h = std::max(int(round(float(resize_h) / 32) * 32), 32);
    resize_w = std::max(int(round(float(resize_w) / 32) * 32), 32);

    cv::resize(img, resize_img, cv::Size(resize_w, resize_h));
    auto ratio_h = float(resize_h) / float(h);
    auto ratio_w = float(resize_w) / float(w);

    input[TASK_RESULT_KEY] = resize_img;
    input["ratio_h"] = ratio_h;
    input["ratio_w"] = ratio_w;
  }
};

IPIPE_REGISTER(Backend, ResizeImgType0Mat, "ResizeImgType0Mat");

class DBNetPost : public PostProcessor<torch::Tensor> {
 private:
  PaddleOCR::PostProcessor post_processor_;
  double det_db_thresh_ = 0.3;
  double det_db_box_thresh_ = 0.5;
  double det_db_unclip_ratio_ = 2;
  bool use_polygon_score_ = false;

 public:
  virtual void forward(std::vector<torch::Tensor> net_outputs, std::vector<dict> input,
                       const std::vector<torch::Tensor>& net_inputs) override {
    if (net_outputs.empty()) return;

    std::vector<std::vector<std::vector<int>>> boxes;

    net_outputs[0] = net_outputs[0].to(torch::kCPU);
    // torch::save(net_outputs[0], "a.pt");
    auto tensor = net_outputs[0].gt(det_db_thresh_) * 255;

    // tensor = tensor.mul(255).clamp(0, 255).to(torch::kByte);
    tensor = tensor.to(torch::kByte).contiguous();
    int64_t height = tensor.size(2);
    int64_t width = tensor.size(3);

    for (std::size_t i = 0; i < input.size(); ++i) {
      cv::Mat original_img = any_cast<cv::Mat>(input[i]->at("original_img"));
      cv::Mat mat =
          cv::Mat(cv::Size(width, height), CV_8UC1, tensor.data_ptr<uchar>() + height * width * i);

      float* prob = net_outputs[0].data_ptr<float>() + height * width * i;
      cv::Mat pred_map = cv::Mat(cv::Size(width, height), CV_32FC1, prob);

      boxes = post_processor_.BoxesFromBitmap(pred_map, mat, this->det_db_box_thresh_,
                                              this->det_db_unclip_ratio_, this->use_polygon_score_);
      float ratio_h = any_cast<float>(input[i]->at("ratio_h"));
      float ratio_w = any_cast<float>(input[i]->at("ratio_w"));
      boxes = post_processor_.FilterTagDetRes(boxes, ratio_h, ratio_w, original_img);

      for (auto& box : boxes) {
      }
      // boxes.clear();
      (*input[i])[TASK_BOX_KEY] = boxes;
      (*input[i])[TASK_RESULT_KEY] = boxes;
      // PaddleOCR::Utility::VisualizeBboxes(original_img, boxes);
    }
    return;

    return PostProcessor<torch::Tensor>::forward(net_outputs, input, net_inputs);
  }
};

IPIPE_REGISTER(PostProcessor<torch::Tensor>, DBNetPost, "dbnet");

class RotatePost : public PostProcessor<torch::Tensor> {
 private:
  float cls_thresh_ = 0.5;

 public:
  virtual void forward(std::vector<torch::Tensor> net_outputs, std::vector<dict> input,
                       const std::vector<torch::Tensor>& net_inputs) override {
    auto tensor = net_outputs[0].to(torch::kCPU);

    for (std::size_t i = 0; i < input.size(); ++i) {
      cv::Mat src_img = any_cast<cv::Mat>(input[i]->at("cropped_img"));
      float prob = tensor[i][1].item<float>();

      if (prob > cls_thresh_) {
        cv::rotate(src_img, src_img, 1);
      }

      (*input[i])[TASK_RESULT_KEY] = src_img;
      (*input[i])["_sort_score"] = float(src_img.cols * 1.0f / (1e-5 + src_img.rows));
    }
    return;
  }
};

IPIPE_REGISTER(PostProcessor<torch::Tensor>, RotatePost, "rotate");

class RecPost : public PostProcessor<torch::Tensor> {
 private:
  std::vector<std::string> label_list_;

 public:
  virtual bool init(const std::unordered_map<std::string, std::string>& config, dict) {
    std::string path = config.at("label_path");

    std::ifstream in(path);
    std::string line;
    if (in) {
      while (getline(in, line)) {
        label_list_.push_back(line);
      }
    } else {
      std::cout << "no such label file: " << path << ", exit the program..." << std::endl;
      return false;
    }

    label_list_.insert(this->label_list_.begin(),
                       "#");  // blank char for ctc
    label_list_.push_back(" ");

    return true;
  };
  virtual void forward(std::vector<torch::Tensor> net_outputs, std::vector<dict> input,
                       const std::vector<torch::Tensor>& net_inputs) override {
    auto argmax_idx = net_outputs[0].argmax(2, true);
    auto max_values = net_outputs[0].gather(2, argmax_idx);
    argmax_idx = argmax_idx.to(torch::kCPU, torch::kInt);
    max_values = max_values.cpu();
    auto* pindex = argmax_idx.data_ptr<int>();
    auto* pvalue = max_values.data_ptr<float>();

    auto shape = net_outputs[0].sizes();

    for (std::size_t batch_index = 0; batch_index != shape[0]; batch_index++) {
      std::vector<std::string> str_res;

      int last_index = 0;
      float score = 0.f;
      int count = 0;
      float max_value = 0.0f;

      for (std::size_t n = 0; n != shape[1]; n++) {
        auto argmax_idx = pindex[shape[1] * batch_index + n];
        auto max_value = pvalue[shape[1] * batch_index + n];
        if (argmax_idx > 0 && (!(n > 0 && argmax_idx == last_index))) {
          score += max_value;
          count += 1;
          str_res.push_back(label_list_[argmax_idx]);
        }
        last_index = argmax_idx;
      }
      score /= count;
      if (isnan(score)) score = 0;
      // for (int i = 0; i < str_res.size(); i++) {
      //   std::cout << str_res[i];
      // }
      (*input[batch_index])[TASK_RESULT_KEY] = str_join(str_res);

      (*input[batch_index])["scores"] = score;
      // std::cout << "\tscore: " << score << std::endl;
    }
  }
};

IPIPE_REGISTER(PostProcessor<torch::Tensor>, RecPost, "RecPost");
}  // namespace ipipe