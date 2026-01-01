#include "omniback/core/any.hpp"
#include "omniback/core/backend.hpp"
#include "omniback/core/reflect.h"

#include <torch/extension.h>
#include <torch/torch.h>

#include <tvm/ffi/function.h>
#include "torchpipe/csrc/helper/torch.hpp"

using omniback::Box;
using omniback::Boxes;

using omniback::dict;

namespace {
dict box2dict(Box box) {
  dict data = std::make_shared<std::unordered_map<std::string, omniback::any>>();
  data->insert_or_assign("id", box.id);
  data->insert_or_assign("score", box.score);
  data->insert_or_assign("x1", box.x1);
  data->insert_or_assign("y1", box.y1);
  data->insert_or_assign("x2", box.x2);
  data->insert_or_assign("y2", box.y2);
  return data;
}

std::vector<dict> boxes2dicts(std::vector<Box> boxes) {
  std::vector<dict> result;
  result.reserve(boxes.size()); // 预分配内存提升性能
  for (const auto& box : boxes) {
    result.push_back(box2dict(box));
  }
  return result;
}
}

float iou(const Box& a, const Box& b) {
  const float inter_x1 = std::max(a.x1, b.x1);
  const float inter_y1 = std::max(a.y1, b.y1);
  const float inter_x2 = std::min(a.x2, b.x2);
  const float inter_y2 = std::min(a.y2, b.y2);

  const float inter_w = std::max(0.0f, inter_x2 - inter_x1);
  const float inter_h = std::max(0.0f, inter_y2 - inter_y1);
  const float inter_area = inter_w * inter_h;

  const float union_area = a.area() + b.area() - inter_area;

  return union_area > 0 ? inter_area / union_area : 0.0f;
}

omniback::Boxes nms(
    const std::vector<omniback::Box>& boxes,
    float iou_threshold,
    bool class_agnostic = false) {
  Boxes result;
  if (boxes.empty())
    return result;

  // Make a copy for sorting
  std::vector<Box> sorted = boxes;
  std::sort(sorted.begin(), sorted.end(), [](const Box& a, const Box& b) {
    return a.score > b.score;
  });

  if (class_agnostic) {
    // Class-agnostic NMS
    std::vector<bool> suppressed(sorted.size(), false);
    for (size_t i = 0; i < sorted.size(); ++i) {
      if (suppressed[i])
        continue;
      result.add(sorted[i]);
      for (size_t j = i + 1; j < sorted.size(); ++j) {
        if (!suppressed[j] && iou(sorted[i], sorted[j]) > iou_threshold) {
          suppressed[j] = true;
        }
      }
    }
  } else {
    // Class-aware NMS (修复: 使用全局索引)
    std::unordered_map<int, std::vector<size_t>> class_map;
    for (size_t i = 0; i < sorted.size(); ++i) {
      class_map[sorted[i].id].push_back(i);
    }

    std::vector<bool> suppressed(sorted.size(), false);
    for (const auto& [_, indices] : class_map) {
      for (size_t i = 0; i < indices.size(); ++i) {
        const size_t idx_i = indices[i];
        if (suppressed[idx_i])
          continue;
        result.add(sorted[idx_i]);

        for (size_t j = i + 1; j < indices.size(); ++j) {
          const size_t idx_j = indices[j];
          if (!suppressed[idx_j] &&
              iou(sorted[idx_i], sorted[idx_j]) > iou_threshold) {
            suppressed[idx_j] = true;
          }
        }
      }
    }
  }

  return result;
}
dict yolo11_post_cpp(
    torch::Tensor& prediction,
    float conf_thres,
    float iou_thres,
    int max_det) {
  TORCH_CHECK(
      prediction.dim() == 3, "预测张量必须是3维 [batch, features, num_boxes]");
  TORCH_CHECK(prediction.size(0) == 1, "仅支持批大小为1");

  auto x = prediction.index({0}).permute({1, 0}); // [2100, 84]

  auto boxes = x.slice(1, 0, 4); // [2100, 4] (cx, cy, w, h)
  auto scores = x.slice(1, 4); // [2100, 80] (类别分数)
  // scores = torch::sigmoid(scores);

  // 1. 置信度阈值过滤
  auto max_kv = scores.max(1);
  auto id = std::get<1>(max_kv);
  auto max_scores = std::get<0>(max_kv); // 每个框的最高类别分数
  auto keep_mask = max_scores >= conf_thres; // 置信度掩码

  id = id.index({keep_mask});
  boxes = boxes.index({keep_mask});
  scores = scores.index({keep_mask})
               .gather(1, id.unsqueeze(1)); // 仅保留最高分数的类别
  scores = scores.cpu().contiguous(); // 过滤后的分数
  boxes = boxes.cpu().contiguous(); // 过滤后的边界框
  id = id.cpu().contiguous();
  auto num_boxes = boxes.size(0);
  float* pboxes = boxes.data_ptr<float>();
  float* pscores = scores.data_ptr<float>();
  int64_t* pid = id.data_ptr<int64_t>();
  omniback::Boxes re_boxes;
  re_boxes.add_batch_cxcywh(pboxes, pscores, pid, num_boxes);
  re_boxes = nms(re_boxes.boxes, iou_thres);
  if (re_boxes.boxes.size() > max_det)
    re_boxes.boxes.resize(max_det); // sorted

  return boxes2dicts(re_boxes.boxes);
}

dict yolo11_post(
    torch::Tensor& prediction,
    float conf_thres,
    float iou_thres,
    int max_det) {
  // 检查输入张量维度
  return yolo11_post_cpp(prediction, conf_thres, iou_thres, max_det);
  }

namespace omniback {
class Yolo11Post : public omniback::BackendOne {
  void impl_init(
      const std::unordered_map<std::string, std::string>& params,
      const dict& options) override {}
  void forward(const dict& io) override {
    torch::Tensor net_out = any_cast<torch::Tensor>(io->at("data"));
    auto result = yolo11_post_cpp(net_out, 0.25, 0.45, 300);

    std::pair<int, int> offset =
        any_cast<std::pair<int, int>>(io->at("offset"));
    float scale = any_cast<float>(io->at("scale"));
    for (auto& box : result.boxes) {
      box.x1 = (box.x1 - offset.first) / scale;
      box.y1 = (box.y1 - offset.second) / scale;
      box.x2 = (box.x2 - offset.first) / scale;
      box.y2 = (box.y2 - offset.second) / scale;
    }

    (*io)["result"] = result;
  }
};
OMNI_REGISTER(omniback::Backend, Yolo11Post);
} // namespace omniback
// PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
//   m.def(
//       "yolo11_post",
//       &yolo11_post,
//       "Non-Maximum Suppression (NMS)",
//       pybind11::arg("prediction"),
//       pybind11::arg("conf_thres") = 0.25f, // 默认值 0.25
//       pybind11::arg("iou_thres") = 0.45f, // 默认值 0.45
//       pybind11::arg("max_det") = 300); // 默认值 300
// }

TVM_FFI_DLL_EXPORT_TYPED_FUNC(yolo11_post, yolo11_post)
