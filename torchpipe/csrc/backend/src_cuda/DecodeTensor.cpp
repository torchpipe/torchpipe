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

#include "DecodeTensor.hpp"

#include <vector>

#include "c10/cuda/CUDAStream.h"
#include "cuda_runtime.h"

#include <ATen/ATen.h>

#include "base_logging.hpp"
#include "nvjpeg.h"
#include "reflect.h"
#include "exception.hpp"
#include "ipipe_utils.hpp"
#include "Backend.hpp"
#include <c10/cuda/CUDACachingAllocator.h>
#include "torch_allocator.hpp"
namespace ipipe {

bool check_nvjpeg_result(nvjpegStatus_t _e) {
  if (_e == NVJPEG_STATUS_SUCCESS) {
    return true;
  } else {
    SPDLOG_WARN("nvjpeg error: {}", int(_e));
    return false;
  }
}

bool DecodeTensor::init(const std::unordered_map<std::string, std::string>& config_param, dict) {
  params_ =
      std::unique_ptr<Params>(new Params({{"color", "rgb"}, {"data_format", "nchw"}}, {}, {}, {}));
  if (!params_->init(config_param)) return false;
  TRACE_EXCEPTION(color_ = params_->at("color"));
  IPIPE_ASSERT(color_ == "rgb" || color_ == "bgr");
  TRACE_EXCEPTION(data_format_ = params_->at("data_format"));
  IPIPE_ASSERT(data_format_ == "nchw" || data_format_ == "hwc");

  auto tmp = at::empty({1, 1}, at::TensorOptions().device(at::kCUDA, -1));
  nvjpegDevAllocator_t dev_allocator = {&dev_malloc, &dev_free};

  nvjpegBackend_t backend = NVJPEG_BACKEND_DEFAULT;

  if (!check_nvjpeg_result(nvjpegCreateEx(backend, &dev_allocator, nullptr, backend, &handle)))
    return false;
  if (!check_nvjpeg_result(nvjpegJpegStateCreate(handle, &state))) return false;

  constexpr int device_padding = 0;
  constexpr int host_padding = 0;
  if (!check_nvjpeg_result(nvjpegSetDeviceMemoryPadding(device_padding, handle))) return false;
  if (!check_nvjpeg_result(nvjpegSetPinnedMemoryPadding(host_padding, handle))) return false;

  return true;
}

inline bool SupportedSubsampling(const nvjpegChromaSubsampling_t& subsampling) {
  switch (subsampling) {
    case NVJPEG_CSS_444:
    case NVJPEG_CSS_440:
    case NVJPEG_CSS_422:
    case NVJPEG_CSS_420:
    case NVJPEG_CSS_411:
    case NVJPEG_CSS_410:
    case NVJPEG_CSS_GRAY:
      return true;
      //        case NVJPEG_CSS_GRAY:
      //        case NVJPEG_CSS_UNKNOWN:
      //            return false;
    default:
      return false;
  }
}

bool decode(const std::string& data, nvjpegHandle_t handle, nvjpegJpegState_t state,
            at::Tensor& image_tensor, const std::string& color, const std::string& data_format) {
  // assert(color == "rgb" || color == "bgr");
  const auto* blob = (const unsigned char*)data.data();
  int nComponents;
  nvjpegChromaSubsampling_t subsampling;
  int widths[NVJPEG_MAX_COMPONENT];
  int heights[NVJPEG_MAX_COMPONENT];

  if (!check_nvjpeg_result(nvjpegGetImageInfo(handle, blob, data.length(), &nComponents,
                                              &subsampling, widths, heights))) {
    SPDLOG_WARN("nvjpegGetImageInfo failed");
    return false;
  }

  if (nComponents == 1) nComponents = 3;
  if (nComponents != 3) {
    SPDLOG_ERROR("Only support channel == 1 or 3, got  {}", nComponents);
    return false;
  }
  if (!SupportedSubsampling(subsampling)) {
    SPDLOG_ERROR("subsampling not supported");
    return false;
  }

  int h = heights[0];
  int w = widths[0];

  size_t image_size = h * w * nComponents;
  constexpr int max_image_size = 5000 * 5000;
  if (max_image_size < h * w || image_size == 0) {
    std::ostringstream ss;
    ss << "image too large or be zero: " << h * w << " , max image size " << max_image_size;
    SPDLOG_ERROR(ss.str());
    return false;
  }

  auto options = at::TensorOptions()
                     .device(at::kCUDA, -1)
                     .dtype(at::kByte)
                     .layout(at::kStrided)
                     .requires_grad(false);
  if (data_format == "nchw") {
    image_tensor = at::empty({1, nComponents, h, w}, options, at::MemoryFormat::Contiguous);
  } else {
    image_tensor = at::empty({h, w, nComponents}, options, at::MemoryFormat::Contiguous);
  }
  auto* image = image_tensor.data_ptr<unsigned char>();

  decltype(NVJPEG_OUTPUT_BGR) target_color;

  if (color == "rgb") {
    if (data_format == "nchw")
      target_color = NVJPEG_OUTPUT_RGB;
    else
      target_color = NVJPEG_OUTPUT_RGBI;
  } else {
    if (data_format == "nchw")
      target_color = NVJPEG_OUTPUT_BGR;
    else
      target_color = NVJPEG_OUTPUT_BGRI;
  }
  nvjpegImage_t nv_image;
  if (data_format == "nchw") {
    for (size_t i = nComponents; i < NVJPEG_MAX_COMPONENT; i++) {
      nv_image.channel[i] = nullptr;
      nv_image.pitch[i] = 0;
    }
    for (size_t i = 0; i < nComponents; i++) {
      nv_image.channel[i] = image + i * w * h;
      nv_image.pitch[i] = w;
    }
  } else {
    for (size_t i = 1; i < NVJPEG_MAX_COMPONENT; i++) {
      nv_image.channel[i] = nullptr;
      nv_image.pitch[i] = 0;
    }
    nv_image.channel[0] = image;
    nv_image.pitch[0] = w * nComponents;
  }

  if (!check_nvjpeg_result(nvjpegDecode(handle, state, blob, data.length(), target_color, &nv_image,
                                        c10::cuda::getCurrentCUDAStream()))) {
    SPDLOG_WARN("nvjpegDecode failed");
    return false;
  }

  return true;
}

void DecodeTensor::forward(const std::vector<dict>& input_dicts) {
  for (auto input_dict : input_dicts) {
    auto& input = *input_dict;

    if (typeid(std::string) != input[TASK_DATA_KEY].type()) {
      SPDLOG_ERROR(std::string("DecodeTensor: not support the input type: ") +
                   c10::demangle(input[TASK_DATA_KEY].type().name()));
      throw std::runtime_error(std::string("DecodeTensor: not support the input type: ") +
                               c10::demangle(input[TASK_DATA_KEY].type().name()));
    }
    std::string data = any_cast<std::string>(input[TASK_DATA_KEY]);

    at::Tensor tensor;

    if (decode(data, handle, state, tensor, color_, data_format_)) {
      input[TASK_RESULT_KEY] = tensor;
      input["color"] = color_;
      input["data_format"] = data_format_;
    } else {
      SPDLOG_DEBUG("decode failed");
    }
  }
}

IPIPE_REGISTER(Backend, DecodeTensor, "DecodeTensor");

}  // namespace ipipe
