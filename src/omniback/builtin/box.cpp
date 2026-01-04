#include <algorithm>
#include <stdexcept>
#include <unordered_map>

#include "omniback/builtin/box.hpp"

namespace omniback {



  // Boxes operations
  void Boxes::add(const Box& box) {
    boxes.push_back(box);
  }

  void Boxes::clear() {
    boxes.clear();
  }

  size_t Boxes::size() const {
    return boxes.size();
  }

  void Boxes::add_xyxy(
      float x1, float y1, float x2, float y2, float score, int id) {
    boxes.push_back({id, score, x1, y1, x2, y2});
  }

  void Boxes::add_cxcywh(
      float cx, float cy, float w, float h, float score, int id) {
    add_xyxy(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2, score, id);
  }

  float Boxes::iou(const Box& a, const Box& b) {
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

  Boxes Boxes::nms(float iou_threshold, bool class_agnostic) const {
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

  void Boxes::add_batch_cxcywh(
      const float* boxes, const float* scores, const int64_t* ids, size_t n) {
    this->boxes.reserve(this->boxes.size() + n);

    for (size_t i = 0; i < n; ++i) {
      const float cx = boxes[i * 4 + 0];
      const float cy = boxes[i * 4 + 1];
      const float w = boxes[i * 4 + 2];
      const float h = boxes[i * 4 + 3];

      add_cxcywh(cx, cy, w, h, scores[i], static_cast<int>(ids[i]));
    }
  }
} // namespace omniback