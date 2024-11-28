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

// 第三方库头文件
#include <torch/torch.h>
// #include "cuda_runtime.h"
// #include "cuda_runtime_api.h"

// #define CLASS_SPECIFIC

using namespace ipipe;

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

// stuff we know about the network and the input/output blobs
template <int NMS_THRESH = 45, int BBOX_CONF_THRESH = 30>
class BatchingPostProcYolox : public TorchPostProcessor {
 public:
  virtual void forward(std::vector<torch::Tensor> net_outputs, std::vector<dict> input,
                       const std::vector<torch::Tensor>& net_inputs) override {
    if (net_outputs.empty()) return;

    auto final_objs = forward_impl(net_outputs, input, net_inputs);

    for (auto i = 0; i < input.size(); ++i) {
      const std::vector<Object>& objects = final_objs[i];
      std::vector<torch::Tensor> boxes;
      for (const auto& obj : objects) {
        auto box = torch::empty({6});
        box[0] = obj.rect.x;
        box[1] = obj.rect.y;
        box[2] = obj.rect.width + obj.rect.x;
        box[3] = obj.rect.height + obj.rect.y;
        box[4] = obj.label;
        box[5] = obj.prob;
        boxes.emplace_back(box);
      }
      (*input[i])[TASK_RESULT_KEY] = boxes;
      (*input[i])[TASK_BOX_KEY] = boxes;
    }
  }

 protected:
  struct Object {
    cv_rect<float> rect;
    int label;
    float prob;
  };

  std::vector<std::vector<Object>> forward_impl(std::vector<torch::Tensor> net_outputs,
                                                std::vector<dict> input,
                                                const std::vector<torch::Tensor>& net_inputs) {
    net_outputs[0] = net_outputs[0].cpu();

    const auto& input_shape = net_inputs[0].sizes().vec();
    auto net_w = net_inputs[0].size(-1);
    auto net_h = net_inputs[0].size(-2);

    const int num_classes = net_outputs[0].size(-1) - 5;
    assert(num_classes > 0);
    std::vector<std::vector<Object>> all_objects;
    for (auto i = 0; i < input.size(); ++i) {
      const float* prob = net_outputs[0][i].data_ptr<float>();
      std::function<std::pair<float, float>(float, float)> inverse_trans =
          any_cast<std::function<std::pair<float, float>(float, float)>>(
              (*input[i])["inverse_trans"]);

      std::vector<Object> objects;

      decode_outputs(prob, objects, inverse_trans, num_classes, net_h, net_w);
      all_objects.push_back(objects);
    }
    return all_objects;
  }

 private:
  struct GridAndStride {
    int grid0;
    int grid1;
    int stride;
  };

  std::vector<GridAndStride> grid_strides;

  void generate_grids_and_stride(const std::vector<int>& strides, unsigned net_h, unsigned net_w) {
    for (auto stride : strides) {
      int num_grid_y = net_h / stride;
      int num_grid_x = net_w / stride;
      for (int g1 = 0; g1 < num_grid_y; g1++) {
        for (int g0 = 0; g0 < num_grid_x; g0++) {
          grid_strides.push_back((GridAndStride){g0, g1, stride});
        }
      }
    }
  }

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

  void generate_yolox_proposals(const float* feat_blob, float prob_threshold,
                                std::vector<Object>& objects, const int num_classes, unsigned net_h,
                                unsigned net_w) {
    if (grid_strides.empty()) {
      generate_grids_and_stride({8, 16, 32}, net_h, net_w);
    }
    const int num_anchors = grid_strides.size();

    for (int anchor_idx = 0; anchor_idx < num_anchors; anchor_idx++) {
      const int basic_pos = anchor_idx * (num_classes + 5);

      float box_objectness = feat_blob[basic_pos + 4];
      for (int class_idx = 0; class_idx < num_classes; class_idx++) {
        float box_cls_score = feat_blob[basic_pos + 5 + class_idx];
        float box_prob = box_objectness * box_cls_score;
        if (box_prob > prob_threshold) {
          // yolox/models/yolo_head.py decode logic
          const int grid0 = grid_strides[anchor_idx].grid0;
          const int grid1 = grid_strides[anchor_idx].grid1;
          const int stride = grid_strides[anchor_idx].stride;
          float x_center = (feat_blob[basic_pos + 0] + grid0) * stride;
          float y_center = (feat_blob[basic_pos + 1] + grid1) * stride;
          float w = exp(feat_blob[basic_pos + 2]) * stride;
          float h = exp(feat_blob[basic_pos + 3]) * stride;
          float x0 = x_center - w * 0.5f;
          float y0 = y_center - h * 0.5f;

          Object obj;
          obj.rect.x = x0;
          obj.rect.y = y0;
          obj.rect.width = w;
          obj.rect.height = h;
          obj.label = class_idx;
          obj.prob = box_prob;

          objects.push_back(obj);
        }

      }  // class loop

    }  // point anchor loop
  }

  void decode_outputs(const float* prob, std::vector<Object>& objects,
                      std::function<std::pair<float, float>(float, float)> inverse_trans,
                      const int num_classes, unsigned net_h, unsigned net_w) {
    // https://raw.githubusercontent.com/shouxieai/tensorRT_Pro/fb9a20c55516b879abfe9ec9316f8c1547780abd/README.md
    std::vector<Object> proposals;
    {
      generate_yolox_proposals(prob, BBOX_CONF_THRESH / 100., proposals, num_classes, net_h, net_w);
    }

    // std::cout << "num of boxes before nms: " << proposals.size()
    //           << std::endl;
    {
      // qsort_descent_inplace(proposals);
      std::sort(proposals.begin(), proposals.end(),
                [](const Object& b1, const Object& b2) { return b1.prob > b2.prob; });
    }

    std::vector<int> picked;
#ifdef CLASS_SPECIFIC
    nms_sorted_bboxes_class_specific(proposals, picked, NMS_THRESH / 100., num_classes);
#else
    nms_sorted_bboxes(proposals, picked, NMS_THRESH / 100.);
#endif

    int count = picked.size();

    // std::cout << "num of boxes: " << count << std::endl;

    objects.resize(count);
    for (int i = 0; i < count; i++) {
      objects[i] = proposals[picked[i]];
      auto tmp = inverse_trans(objects[i].rect.x + 1, objects[i].rect.y + 1);
      // adjust offset to original unpadded
      float x0 = tmp.first;
      float y0 = tmp.second;
      tmp = inverse_trans(objects[i].rect.x + objects[i].rect.width,
                          objects[i].rect.y + objects[i].rect.height);
      float x1 = tmp.first;
      float y1 = tmp.second;

      // clip
      // x0 = std::max(std::min(x0, (float)(img_w - 1)), 0.f);
      // y0 = std::max(std::min(y0, (float)(img_h - 1)), 0.f);
      // x1 = std::max(std::min(x1, (float)(img_w - 1)), 0.f);
      // y1 = std::max(std::min(y1, (float)(img_h - 1)), 0.f);

      objects[i].rect.x = x0;
      objects[i].rect.y = y0;
      objects[i].rect.width = x1 - x0;
      objects[i].rect.height = y1 - y0;
    }
  }
};

class BatchingPostProcYolox_custom : public BatchingPostProcYolox<20, 40> {
 public:
  void forward(std::vector<torch::Tensor> net_outputs, std::vector<dict> input,
               const std::vector<torch::Tensor>& net_inputs) override {
    if (net_outputs.empty()) return;

    auto final_objs = forward_impl(net_outputs, input, net_inputs);

    for (auto i = 0; i < input.size(); ++i) {
      const std::vector<Object>& objects = final_objs[i];
      std::vector<torch::Tensor> boxes;
      std::vector<torch::Tensor> boxes_for_crop;
      for (const auto& obj : objects) {
        auto box = torch::empty({6});
        box[0] = obj.rect.x;
        box[1] = obj.rect.y;
        box[2] = obj.rect.width + obj.rect.x;
        box[3] = obj.rect.height + obj.rect.y;
        box[4] = obj.label;
        box[5] = obj.prob;
        boxes.emplace_back(box);

        torch::Tensor img = any_cast<torch::Tensor>((*input[i])[TASK_DATA_KEY]);
        auto img_w = img.size(1);
        auto img_h = img.size(0);

        float pad = 1 * std::min(obj.rect.width, obj.rect.height);
        pad = std::min(std::max(pad, 5.f), 40.f);

        auto box_for_crop = torch::empty({6});
        box_for_crop[0] = std::max(0.f, obj.rect.x - pad);
        box_for_crop[1] = std::max(0.f, obj.rect.y - pad);
        box_for_crop[2] = std::min(obj.rect.width + obj.rect.x + pad, img_w - 1.f);
        box_for_crop[3] = std::min(obj.rect.height + obj.rect.y + pad, img_h - 1.f);
        box_for_crop[4] = obj.label;
        box_for_crop[5] = obj.prob;
        boxes_for_crop.emplace_back(box_for_crop);
      }
      (*input[i])[TASK_RESULT_KEY] = boxes;
      (*input[i])[TASK_BOX_KEY] = boxes_for_crop;
    }
  }
};

using BatchingPostProcYolox_default = BatchingPostProcYolox<>;
IPIPE_REGISTER(TorchPostProcessor, BatchingPostProcYolox_default, "BatchingPostProcYolox");

using BatchingPostProcYolox_45_30 = BatchingPostProcYolox<45, 30>;
IPIPE_REGISTER(TorchPostProcessor, BatchingPostProcYolox_45_30, "BatchingPostProcYolox_45_30");

///   v2
/// https://github.com/shouxieai/tensorRT_Pro/blob/2c5db6a987be9ab92abdc78db3ce4305053364f0/src/application/app_yolo/yolo_decode.cu
// https://github.com/shouxieai/tensorRT_Pro/blob/main/README.zh-cn.md
///   v3
///   https://github.com/ttanzhiqiang/onnx_tensorrt_project/blob/main/src/yolox/yolox_detector.cpp
