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

#pragma once
#include <cstdint>
#include <stdexcept>

namespace ipipe {
class SerializedMat {
 public:
  SerializedMat(uint32_t hh, uint32_t ww, uint32_t cc, char* in_data, uint32_t in_len)
      : h(hh), w(ww), c(cc), data(in_data), len(in_len) {
    elemSize = in_len / (h * w * c);
    if (!in_data || (elemSize != 1 && elemSize != 3)) {
      throw std::runtime_error("wrong SerializedMat");
    }
  }
  uint32_t h;
  uint32_t w;
  uint32_t c;
  uint32_t elemSize;
  char* data{nullptr};
  uint32_t len = 0;
};
};  // namespace ipipe