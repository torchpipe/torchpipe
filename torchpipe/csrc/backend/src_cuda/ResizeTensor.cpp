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

#include "ResizeTensor.hpp"

#include <ATen/ATen.h>
// #include <cuda.h>
#include <cuda_runtime.h>
#include <nppcore.h>
// #include <nppi.h>
#include <fstream>
#include <numeric>
#include <thread>
#include "base_logging.hpp"
#include "c10/cuda/CUDAStream.h"
#include "hw_batching.hpp"
#include "nppi_geometry_transforms.h"
#include "reflect.h"
#include "torch_utils.hpp"
#include "ipipe_utils.hpp"
#include "exception.hpp"

namespace ipipe {
at::Tensor nppiresize_topleft(at::Tensor& input, int target_h, int target_w, int pad_value) {
  if (input.scalar_type() != at::kByte && input.scalar_type() != at::kFloat) {
    throw std::invalid_argument("error type: need scalar type: kByte or kFloat");
  }

  int img_h = input.sizes()[0];
  int img_w = input.sizes()[1];
  int c = input.sizes()[2];

  auto single_scale = std::max(float(img_w) / float(target_w), float(img_h) / float(target_h));
  int true_w = floor(img_w / single_scale);
  int true_h = img_h / single_scale;
  auto options = at::TensorOptions()
                     .device(at::kCUDA, -1)
                     .dtype(input.scalar_type())  // at::kByte
                     .layout(at::kStrided)
                     .requires_grad(false);
  auto output_tensor = at::full({target_h, target_w, c}, int(pad_value), options);
  if (!output_tensor.is_contiguous()) {
    output_tensor = output_tensor.contiguous();
  }

  auto* output = output_tensor.data_ptr();

  if (!input.is_contiguous()) input = input.contiguous();
  void* image = input.data_ptr();

  // typedef enum {
  //     NPPI_INTER_UNDEFINED = 0,
  //     NPPI_INTER_NN = 1,          /**<  最近邻插值 */
  //     NPPI_INTER_LINEAR = 2,      /**<  线性插值 */
  //     NPPI_INTER_CUBIC = 4,       /**<  三次插值 */
  //     NPPI_INTER_CUBIC2P_BSPLINE, /**<  Two-parameter cubic filter (B=1,
  //     C=0)
  //                                  */
  //     NPPI_INTER_CUBIC2P_CATMULLROM, /**<  Two-parameter cubic filter (B=0,
  //                                       C=1/2) */
  //     NPPI_INTER_CUBIC2P_B05C03,     /**<  Two-parameter cubic filter
  //     (B=1/2,
  //                                       C=3/10) */
  //     NPPI_INTER_SUPER = 8,          /**<  Super sampling. */
  //     NPPI_INTER_LANCZOS = 16,       /**<  Lanczos filtering. */
  //     NPPI_INTER_LANCZOS3_ADVANCED =
  //         17, /**<  Generic Lanczos filtering with order 3. */
  //     NPPI_SMOOTH_EDGE = (1 << 31) /**<  Smooth edge filtering. */
  // } NppiInterpolationMode;

  NppiInterpolationMode eInterploationMode = NPPI_INTER_CUBIC;
  if (target_w < img_w && target_h < img_h) {
    eInterploationMode = NPPI_INTER_SUPER;
    // https://forums.developer.nvidia.com/t/npp-library-functions-nppiresize-8u-c3r-and-nppibgrtolab-8u-c3r-differ-from-cv-resize-output/66608/7
  }

  NppiSize image_a_size = {.width = img_w, .height = img_h};
  NppiRect image_a_roi = {.x = 0, .y = 0, .width = img_w, .height = img_h};

  NppiSize image_b_size = {.width = target_w, .height = target_h};
  NppiRect image_b_roi = {.x = 0, .y = 0, .width = true_w, .height = true_h};

  NppStatus result;
  if (input.scalar_type() != at::kByte) {
    result = nppiResize_8u_C3R((Npp8u*)image, img_w * 3, image_a_size, image_a_roi, (Npp8u*)output,
                               target_w * 3, image_b_size, image_b_roi, eInterploationMode);
  } else {
    result = nppiResize_32f_C3R((Npp32f*)image, img_w * 3 * sizeof(Npp32f), image_a_size,
                                image_a_roi, (Npp32f*)output, target_w * 3 * sizeof(Npp32f),
                                image_b_size, image_b_roi, eInterploationMode);
  }

  if (result != NPP_SUCCESS) {
    std::cout << "nppiresize_topleft Error executing Resize -- code: " << result << " " << target_w
              << " " << img_w << " " << target_h << " " << img_h << std::endl;
    return at::Tensor();
  }

  return output_tensor;
}

at::Tensor nppiresize(at::Tensor input, int target_h, int target_w) {
  int num = input.sizes().size() == 4 ? input.sizes()[0] : 1;
  if (num != 1 || input.size(-1) != 3) {
    throw std::invalid_argument("nppiresize ;: error num or input.size" +
                                std::to_string(input.size(0)) + std::to_string(input.size(1)) +
                                std::to_string(input.size(2)) + std::to_string(input.size(-1)));
  }
  if (input.scalar_type() != at::kByte && input.scalar_type() != at::kFloat) {
    throw std::invalid_argument("error datatype of tensor.  need datatype float or char.");
  }

  int img_h = input.sizes()[0];
  int img_w = input.sizes()[1];
  int c = input.sizes()[2];

  auto options = at::TensorOptions()
                     .device(at::kCUDA, -1)
                     .dtype(input.scalar_type())  // at::kByte
                     .layout(at::kStrided)
                     .requires_grad(false);

  auto output_tensor = at::full({target_h, target_w, c}, 0, options);
  auto* output = output_tensor.data_ptr();

  if (!input.is_contiguous()) input = input.contiguous();
  void* image = input.data_ptr();

  NppiInterpolationMode eInterploationMode = NPPI_INTER_CUBIC;
  if (target_w < img_w && target_h < img_h) {
    eInterploationMode = NPPI_INTER_SUPER;
    // https://forums.developer.nvidia.com/t/npp-library-functions-nppiresize-8u-c3r-and-nppibgrtolab-8u-c3r-differ-from-cv-resize-output/66608/7
  }

  // eInterploationMode = NPPI_INTER_LINEAR;
  NppiSize image_a_size = {.width = img_w, .height = img_h};
  NppiRect image_a_roi = {.x = 0, .y = 0, .width = img_w, .height = img_h};

  NppiSize image_b_size = {.width = target_w, .height = target_h};
  NppiRect image_b_roi = {.x = 0, .y = 0, .width = target_w, .height = target_h};
  NppStatus result;
  if (input.scalar_type() == at::kByte) {
    result = nppiResize_8u_C3R((Npp8u*)image, img_w * 3, image_a_size, image_a_roi, (Npp8u*)output,
                               target_w * 3, image_b_size, image_b_roi, eInterploationMode);

  } else {
    result = nppiResize_32f_C3R((Npp32f*)image, img_w * 3 * sizeof(Npp32f), image_a_size,
                                image_a_roi, (Npp32f*)output, target_w * 3 * sizeof(Npp32f),
                                image_b_size, image_b_roi, eInterploationMode);
  }
  if (result != NPP_SUCCESS) {
    SPDLOG_ERROR(
        "nppiresize: Error executing Resize -- code: {} img_w {} img_h {} target_w {} target_h {}",
        result, img_w, img_h, target_w, target_h);
    throw std::runtime_error("nppiresize falied");
  }

  return output_tensor;
}

bool ResizeTensor::init(const std::unordered_map<std::string, std::string>& config_param,
                        dict dict_config) {
  params_ = std::unique_ptr<Params>(new Params({}, {"resize_h", "resize_w"}, {}, {}));
  if (!params_->init(config_param)) return false;

  TRACE_EXCEPTION(resize_h_ = std::stoi(params_->operator[]("resize_h")));
  TRACE_EXCEPTION(resize_w_ = std::stoi(params_->operator[]("resize_w")));
  if (resize_h_ > 1024 * 1024 || resize_w_ > 1024 * 1024 || resize_h_ < 1 || resize_w_ < 1 ||
      resize_w_ * resize_h_ > 1024 * 1024 * 100) {
    SPDLOG_ERROR("ResizeTensor: illigle h or w: h=" + std::to_string(resize_h_) +
                 "w=" + std::to_string(resize_w_));
    return false;
  }

  return true;
}

void ResizeTensor::forward(dict input_dict) {
  auto input_tensor = dict_get<at::Tensor>(input_dict, TASK_DATA_KEY);

  bool is_hwc_tensor = is_hwc(input_tensor);

  input_tensor = img_1chw_guard(input_tensor).to(at::kFloat);
  if (!input_tensor.is_contiguous()) input_tensor = input_tensor.contiguous();

  at::Tensor im_resize;

  if (input_tensor.size(2) == resize_h_ && input_tensor.size(3) == resize_w_) {
    im_resize = input_tensor;
  } else {
    im_resize = at::upsample_bilinear2d(input_tensor.to(at::kFloat), {resize_h_, resize_w_}, true);
  }
  if (is_hwc_tensor) {
    im_resize = im_resize.permute({0, 2, 3, 1}).squeeze(0);
  }

  (*input_dict)[TASK_RESULT_KEY] = im_resize;
}

IPIPE_REGISTER(Backend, ResizeTensor, "ResizeTensor");

bool ResizeTensorV1::init(const std::unordered_map<std::string, std::string>& config_param,
                          dict dict_config) {
  params_ = std::unique_ptr<Params>(
      new Params({{"instance_num", "1"}}, {"resize_h", "resize_w"}, {}, {}));
  if (!params_->init(config_param)) return false;
  int instance_num;
  TRACE_EXCEPTION(instance_num = std::stoi(params_->at("instance_num")));
  IPIPE_ASSERT(instance_num == 1);
  // 对于nppi库，全局只有唯一流，这样多实例可能会有问题，等待进一步探索，暂时只支持实例数目为1

  TRACE_EXCEPTION(resize_h_ = std::stoi(params_->operator[]("resize_h")));
  TRACE_EXCEPTION(resize_w_ = std::stoi(params_->operator[]("resize_w")));
  if (resize_h_ > 1024 * 1024 || resize_w_ > 1024 * 1024 || resize_h_ < 1 || resize_w_ < 1 ||
      resize_w_ * resize_h_ > 1024 * 1024 * 100) {
    SPDLOG_ERROR("ResizeTensor: illigle h or w: h=" + std::to_string(resize_h_) +
                 "w=" + std::to_string(resize_w_));
    return false;
  }

  return true;
}

//
// 新增的TensorResize操作，保证输入与输出类型一致
// 1.对于gpu上tensor，如果是float，则使用at::upsample_bilinear2d进行resize，返回float
// 2.对于gpu上tensor，如果是uint8，则使用nppiresize，返回uint8
// 3.对于cpu上tensor，默认使用at::upsample_bilinear2d，根据输入tensor类型，返回类型一致。

void ResizeTensorV1::forward(dict input_dict) {
  auto input_tensor = dict_get<at::Tensor>(input_dict, TASK_DATA_KEY);
  at::Tensor im_resize;

  if (input_tensor.scalar_type() == at::kFloat || (!input_tensor.is_cuda())) {
    input_tensor = img_1chw_guard(input_tensor);
    if (!input_tensor.is_contiguous()) input_tensor = input_tensor.contiguous();

    if (input_tensor.size(2) == resize_h_ && input_tensor.size(3) == resize_w_) {
      im_resize = input_tensor;
    } else {
      im_resize =
          at::upsample_bilinear2d(input_tensor.to(at::kFloat), {resize_h_, resize_w_}, true);
    }
    // 如果输入是kByte，做强制转换！
    if (input_tensor.scalar_type() == at::kByte) {
      im_resize = im_resize.to(at::kByte);
    }

  } else if (input_tensor.scalar_type() == at::kByte) {
    input_tensor = img_hwc_guard(input_tensor);
    if (!input_tensor.is_contiguous()) input_tensor = input_tensor.contiguous();

    if (input_tensor.size(0) == resize_h_ && input_tensor.size(1) == resize_w_) {
      im_resize = input_tensor;
    } else {
      // 确保与torch的流同步
      cudaStream_t curr_stream = c10::cuda::getCurrentCUDAStream();
      if (nppGetStream() != curr_stream) {
        nppSetStream(curr_stream);
      }
      im_resize = nppiresize(input_tensor, resize_h_, resize_w_);
    }
    // 转换到1chw
    im_resize = img_1chw_guard(im_resize);
  } else {
    SPDLOG_ERROR("error: ResizeTensorV1 backend error, need scalar type: kByte or kFloat!!");
  }

  (*input_dict)[TASK_RESULT_KEY] = im_resize;
}

IPIPE_REGISTER(Backend, ResizeTensorV1, "ResizeTensorV1");

}  // namespace ipipe