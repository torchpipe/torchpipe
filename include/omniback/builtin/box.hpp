#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace omniback {

/**
 * @brief Bounding box structure in XYXY format
 */
struct Box {
  int id; // Class ID
  float score; // Confidence score
  float x1, y1, x2, y2; // Coordinates in XYXY format

  // Computed properties
  float cx() const {
    return (x1 + x2) / 2;
  }
  float cy() const {
    return (y1 + y2) / 2;
  }
  float width() const {
    return x2 - x1;
  }
  float height() const {
    return y2 - y1;
  }
  float area() const {
    return width() * height();
  }
};

/**
 * @brief Container for bounding boxes with advanced operations
 */
class Boxes {
 public:
  std::vector<Box> boxes;

  // Core operations
  void add(const Box& box);
  void clear();
  size_t size() const;

  // Batch operations
  void add_xyxy(float x1, float y1, float x2, float y2, float score, int id);
  void add_cxcywh(float cx, float cy, float w, float h, float score, int id);

  /**
   * @brief Add batch of boxes in CXCYWH format
   * @param boxes Array of shape [n,4] in (cx,cy,w,h) order
   * @param scores Array of shape [n] with confidence scores
   * @param ids Array of shape [n] with class IDs
   * @param n Number of boxes
   */
  void add_batch_cxcywh(
      const float* boxes,
      const float* scores,
      const int64_t* ids,
      size_t n);

  // Metrics
  static float iou(const Box& a, const Box& b);

  // Algorithms
  Boxes nms(float iou_threshold, bool class_agnostic = false) const;
};

} // namespace omniback