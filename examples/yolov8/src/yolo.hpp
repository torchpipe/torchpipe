// from
// https://github.com/Monday-Leo/YOLOv8_Tensorrt/blob/3fe31e0167834c21f18c18502ce3c1b9675b0b2a/yolo.hpp#L17
#ifndef __YOLO_HPP__
#define __YOLO_HPP__

#include <future>
#include <memory>
#include <string>
#include <vector>

namespace yolo {

enum class Type : int {
  V5 = 0,
  X = 1,
  V3 = 2,
  V7 = 3,
  V8 = 5,
  V8Seg = 6  // yolov8 instance segmentation
};

const char *type_name(Type type);

};  // namespace yolo

#endif  // __YOLO_HPP__