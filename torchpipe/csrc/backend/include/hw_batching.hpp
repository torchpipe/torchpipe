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

#pragma once

#include <vector>

#include "Backend.hpp"
#include "dict.hpp"
namespace ipipe {

struct HW {
  /* data */
  int h;
  int w;
};

/* 多个矩形（data）保持长宽比缩小到 不大于max_shape之后，求data与
 * min_shape的最小外接矩形
 *
 *
 *
 */
inline HW max_batching(const std::vector<HW>& data, HW min_shape, HW max_shape, HW align) {
  //  step one: less than max shape
  for (std::size_t i = 0; i < data.size(); ++i) {
    const auto h_ratio = data[i].h * 1.0 / max_shape.h;
    const auto w_ratio = data[i].w * 1.0 / max_shape.w;
    if (h_ratio > 1. || w_ratio > 1.) {
      if (h_ratio > w_ratio) {
        min_shape.h = std::max(min_shape.h, max_shape.h);
        min_shape.w = std::max(min_shape.w, int(data[i].w / h_ratio));

      } else {
        min_shape.h = std::max(min_shape.h, int(data[i].h / w_ratio));
        min_shape.w = std::max(min_shape.w, max_shape.w);
      }
    } else {
      min_shape.h = std::max(min_shape.h, data[i].h);
      min_shape.w = std::max(min_shape.w, data[i].w);
    }
  }

  //  step two: align
  if (min_shape.h % align.h != 0) {
    min_shape.h += align.h - min_shape.h % align.h;
  }
  if (min_shape.w % align.w != 0) {
    min_shape.w += align.w - min_shape.w % align.w;
  }

  // 防止浮点运算的误差产生越界, assume min_shape and max_shape aligned with
  // align
  min_shape.h = std::min(min_shape.h, max_shape.h);
  min_shape.w = std::min(min_shape.w, max_shape.w);

  assert(min_shape.h % align.h == 0);
  assert(min_shape.w % align.w == 0);

  return min_shape;
};

};  // namespace ipipe
