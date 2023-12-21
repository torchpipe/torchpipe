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

// modified from https://github.com/cyrusbehr/YOLOv8-TensorRT-CPP/blob/main/src/yolov8.cpp
#include <cassert>
#include <chrono>
#include <fstream>
#include <iostream>
#include <mutex>
#include <numeric>
#include <sstream>
#include <vector>

// 本地项目头文件
#include "any.hpp"
#include "dict.hpp"
#include "reflect.h"
#include "prepost.hpp"
#include "torchpipe/extension.h"

// 第三方库头文件
#include <ATen/ATen.h>

// #define CLASS_SPECIFIC

constexpr float NMS_THRESH = 0.45;
constexpr float BBOX_CONF_THRESH = 0.25;

constexpr int TOP_K = 100;

namespace {  // nms

template <typename T>
class cv_rect {
 public:
  cv_rect() : x(0), y(0), width(0), height(0) {}

  T area() const { return width * height; };

  T x;
  T y;
  T width;
  T height;
};

template <typename T>
static inline cv_rect<T> operator&(const cv_rect<T>& a, const cv_rect<T>& b) {
  cv_rect<T> c;
  T x1 = std::max(a.x, b.x);
  T y1 = std::max(a.y, b.y);
  c.width = std::min(a.x + a.width, b.x + b.width) - x1;
  c.height = std::min(a.y + a.height, b.y + b.height) - y1;
  c.x = x1;
  c.y = y1;
  if (c.width <= 0 || c.height <= 0) c = cv_rect<T>();
  return c;
}

struct Object {
  Object() = default;

  cv_rect<float> rect;
  int label;
  float prob;
};

float intersection_area(const Object& a, const Object& b) {
  cv_rect<float> inter = a.rect & b.rect;
  return inter.area();
}

void nms_sorted_bboxes(const std::vector<Object>& faceobjects, std::vector<int>& picked,
                       float nms_threshold) {
  picked.clear();

  const int n = faceobjects.size();

  std::vector<float> areas(n);
  for (int i = 0; i < n; i++) {
    areas[i] = faceobjects[i].rect.area();
  }

  for (int i = 0; i < n; i++) {
    const Object& a = faceobjects[i];

    int keep = 1;
    for (int j = 0; j < (int)picked.size(); j++) {
      const Object& b = faceobjects[picked[j]];

      // intersection over union
      float inter_area = intersection_area(a, b);
      float union_area = areas[i] + areas[picked[j]] - inter_area;
      // float IoU = inter_area / union_area
      if (inter_area / union_area > nms_threshold) keep = 0;
    }

    if (keep) picked.push_back(i);
  }
}

void nms_sorted_bboxes_target(const std::vector<Object>& faceobjects, std::vector<int>& picked,
                              float nms_threshold, int index_classes,
                              const std::vector<float>& areas) {
  picked.clear();

  const int n = faceobjects.size();

  for (int i = 0; i < n; i++) {
    const Object& a = faceobjects[i];
    if (a.label != index_classes) continue;

    int keep = 1;
    for (int j = 0; j < (int)picked.size(); j++) {
      const Object& b = faceobjects[picked[j]];

      // intersection over union
      float inter_area = intersection_area(a, b);
      float union_area = areas[i] + areas[picked[j]] - inter_area;
      // float IoU = inter_area / union_area
      if (inter_area / union_area > nms_threshold) keep = 0;
    }

    if (keep) picked.push_back(i);
  }
}

void nms_sorted_bboxes_class_specific(const std::vector<Object>& faceobjects,
                                      std::vector<int>& picked, float nms_threshold,
                                      int num_classes) {
  const int n = faceobjects.size();

  std::vector<float> areas(n);
  for (int i = 0; i < n; i++) {
    areas[i] = faceobjects[i].rect.area();
  }
  for (std::size_t i = 0; i < num_classes; ++i) {
    std::vector<int> local_picked;
    nms_sorted_bboxes_target(faceobjects, local_picked, nms_threshold, i, areas);
    picked.insert(picked.end(), local_picked.begin(), local_picked.end());
  }
}
}  // namespace
using namespace ipipe;

class PostProcYoloV8 : public Backend {
 public:
  virtual bool init(const std::unordered_map<std::string, std::string>& config, dict dict_config) {
    if (config.find("net_h") == config.end() || config.find("net_w") == config.end()) {
      std::cerr << "net_h or net_w not found" << std::endl;
      return false;
    }
    net_h_ = std::stoi(config.at("net_h"));
    net_w_ = std::stoi(config.at("net_w"));
    return true;
  };
  void forward(const std::vector<dict>& data) override {
    IPIPE_ASSERT(data.size() == 1);
    at::Tensor result = dict_get<at::Tensor>(data[0], TASK_DATA_KEY);
    forward({result}, data);
  }

 private:
  void forward(std::vector<at::Tensor> net_outputs, std::vector<dict> input) {
    if (net_outputs.empty()) return;

    auto final_objs = forward_impl(net_outputs, input);

    for (auto i = 0; i < input.size(); ++i) {
      const std::vector<Object>& objects = final_objs[i];

      std::vector<std::vector<int>> boxes(objects.size());
      std::vector<int> labels(objects.size());
      std::vector<float> probs(objects.size());
      for (std::size_t j = 0; j < objects.size(); ++j) {
        const auto& obj = objects[j];
        std::vector<int> x1y1x2y2{obj.rect.x, obj.rect.y, obj.rect.width + obj.rect.x,
                                  obj.rect.height + obj.rect.y};

        labels[j] = obj.label;
        probs[j] = obj.prob;
        boxes[j] = x1y1x2y2;
      }

      (*input[i])[TASK_RESULT_KEY] = boxes;
      (*input[i])[TASK_BOX_KEY] = boxes;
      (*input[i])["labels"] = labels;
      (*input[i])["probs"] = probs;
    }
  }

 protected:
  std::vector<std::vector<Object>> forward_impl(std::vector<at::Tensor> net_outputs,
                                                std::vector<dict> input) {
    const int numChannels = net_outputs[0].size(1);
    const int numClasses = numChannels - 4;
    assert(numClasses > 0);

    auto numAnchors = net_outputs[0].size(2);

    if (!is_cpu_tensor(net_outputs[0]))  // cpu tensor to gpu tensor
      net_outputs[0] = net_outputs[0].permute({0, 2, 1}).contiguous().cpu();
    else {
      net_outputs[0] = net_outputs[0].permute({0, 2, 1}).contiguous();
    }

    assert(net_outputs[0].is_contiguous());

    float* prob = net_outputs[0][0].data_ptr<float>();

    // float ratio = dict_get<float>(input[0], "ratio");

    auto net_w = net_w_;
    auto net_h = net_h_;
    at::Tensor img = dict_get<at::Tensor>(input[0], "other");
    int offset_size = img.sizes().size() == 4 ? 2 : 0;
    auto img_h = static_cast<float>(img.size(offset_size));
    auto img_w = static_cast<float>(img.size(offset_size + 1));

    float ratio =
        1.f / std::min(net_w / static_cast<float>(img_w), net_h / static_cast<float>(img_h));

    std::vector<Object> bboxes;
    std::vector<int> indices;

    // cv::Mat output = cv::Mat(numAnchors, numChannels, CV_32F, prob);
    // output = output.t();
    // assert(output.isContinuous());

    // Get all the YOLO proposals
    for (int i = 0; i < numAnchors; i++) {
      auto rowPtr = prob + i * numChannels;  // output.row(i).ptr<float>();
      auto bboxesPtr = rowPtr;
      auto scoresPtr = rowPtr + 4;
      auto maxSPtr = std::max_element(scoresPtr, scoresPtr + numClasses);
      float score = *maxSPtr;
      if (score > BBOX_CONF_THRESH) {
        float x = *bboxesPtr++;
        float y = *bboxesPtr++;
        float w = *bboxesPtr++;
        float h = *bboxesPtr;

        float x0 = std::clamp((x - 0.5f * w) * ratio, 0.f, img_w);
        float y0 = std::clamp((y - 0.5f * h) * ratio, 0.f, img_h);
        float x1 = std::clamp((x + 0.5f * w) * ratio, 0.f, img_w);
        float y1 = std::clamp((y + 0.5f * h) * ratio, 0.f, img_h);

        int label = maxSPtr - scoresPtr;
        Object obj;
        obj.rect.x = x0;
        obj.rect.y = y0;
        obj.rect.width = x1 - x0;
        obj.rect.height = y1 - y0;
        obj.label = label;
        obj.prob = score;
        bboxes.push_back(obj);
      }
    }

    std::sort(bboxes.begin(), bboxes.end(),
              [](const Object& b1, const Object& b2) { return b1.prob > b2.prob; });

    std::vector<int> picked;
#ifdef CLASS_SPECIFIC
    nms_sorted_bboxes_class_specific(bboxes, picked, NMS_THRESH, num_classes);
#else
    nms_sorted_bboxes(bboxes, picked, NMS_THRESH);
#endif

    int count = picked.size();

    // std::cout << "num of boxes: " << count << std::endl;

    std::vector<Object> objects;

    // Run NMS

    // Choose the top k detections
    int cnt = 0;
    for (auto& chosenIdx : picked) {
      if (cnt >= TOP_K) {
        break;
      }

      objects.push_back(bboxes[chosenIdx]);

      cnt += 1;
    }

    return {objects};
  }

 private:
  int net_h_;
  int net_w_;
};
// using PostProcYoloV8_45_30 = PostProcYoloV8;
// IPIPE_REGISTER(PostProcessor<at::Tensor>, PostProcYoloV8_45_30, "PostProcYoloV8_45_30");
IPIPE_REGISTER(Backend, PostProcYoloV8, "PostProcYoloV8");

namespace ipipe {
class FilterScore : public Filter {
 public:
  status forward(dict data) override {
    constexpr auto thres = 0.3;
    auto iter = data->find("score");
    if (iter == data->end()) {
      return status::Error;
    }
    float score = any_cast<float>(iter->second);
    if (score < thres) {
      // if ((float)rand() / RAND_MAX < 0.3) {
      return status::Run;
    } else {
      return status::Skip;
    }
  }
};
IPIPE_REGISTER(Filter, FilterScore, "filter_score");
}  // namespace ipipe