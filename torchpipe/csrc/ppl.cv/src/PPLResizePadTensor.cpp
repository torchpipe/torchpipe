
#include "PPLResizePadTensor.hpp"

#include <ppl/cv/cuda/resize.h>
// #include <torch/torch.h>
#include "c10/cuda/CUDAStream.h"

#include <ATen/ATen.h>

#include "ipipe_utils.hpp"
#include "base_logging.hpp"
#include "reflect.h"
#include "exception.hpp"
#include "ppl_cv_helper.hpp"
#include "torch_utils.hpp"
namespace ipipe {
bool PPLResizeCenterPadTensor::init(const std::unordered_map<std::string, std::string>& config,
                                    dict dict_config) {
  throw std::runtime_error("not implemented");
  params_ = std::unique_ptr<Params>(new Params({{"resize_h", "0"}, {"resize_w", "0"}}, {}, {}, {}));
  if (!params_->init(config)) return false;

  TRACE_EXCEPTION(resize_h_ = std::stoi(params_->operator[]("resize_h")));
  TRACE_EXCEPTION(resize_w_ = std::stoi(params_->operator[]("resize_w")));
  return true;
}

void PPLResizeCenterPadTensor::forward(dict input_dict) {
  try_update(input_dict, "resize_h", resize_h_);
  try_update(input_dict, "resize_w", resize_w_);
  IPIPE_ASSERT(resize_h_ > 0 && resize_w_ > 0 && resize_h_ * resize_w_ < 1024 * 1024);

  HWCTensorWrapper ppl(input_dict, resize_h_, resize_w_, true);
  auto& input_tensor = ppl.input_tensor;
  auto& output_tensor = ppl.output_tensor;

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  int img_h = input_tensor.sizes()[0];
  int img_w = input_tensor.sizes()[1];
  int c = input_tensor.sizes()[2];
  IPIPE_ASSERT(c == 3);

  float ratio =
      std::min(resize_h_ / static_cast<float>(img_h), resize_w_ / static_cast<float>(img_w));
  int target_h = static_cast<int>(img_h * ratio);
  int target_w = static_cast<int>(img_w * ratio);
  IPIPE_ASSERT(target_h <= resize_h_ && target_w <= resize_w_);
  int top = (resize_h_ - target_h) / 2;
  int bottom = resize_h_ - target_h - top;
  int left = (resize_w_ - target_w) / 2;
  int right = resize_w_ - target_w - left;

  if (input_tensor.scalar_type() == at::kByte) {
    unsigned char* image = input_tensor.data_ptr<unsigned char>();
    unsigned char* output = output_tensor.data_ptr<unsigned char>();
    output += top * output_tensor.stride(0) + left * c;

    assert(resize_w_ * c == output_tensor.stride(0));
    assert(img_w * c == input_tensor.stride(0));
    auto ret = ppl::cv::cuda::Resize<unsigned char, 3>(
        stream, img_h, img_w, input_tensor.stride(0), image, target_h, target_w,
        output_tensor.stride(0), output, ppl::cv::INTERPOLATION_LINEAR);
    if (ret != ppl::common::RC_SUCCESS) {
      SPDLOG_ERROR("PPLResizeCenterPadTensor: Resize failed, ret = {}", ret);
      return;
    }
  } else if (input_tensor.scalar_type() == at::kFloat) {
    float* image = input_tensor.data_ptr<float>();
    float* output = output_tensor.data_ptr<float>();
    output += top * output_tensor.stride(0) + left * c;

    auto ret = ppl::cv::cuda::Resize<float, 3>(stream, img_h, img_w, input_tensor.stride(0), image,
                                               target_h, target_w, output_tensor.stride(0), output,
                                               ppl::cv::INTERPOLATION_LINEAR);
    if (ret != ppl::common::RC_SUCCESS) {
      SPDLOG_ERROR("PPLResizeCenterPadTensor: Resize failed");
      return;
    }
  } else {
    throw std::invalid_argument("Only support type of float or char.");
  }
  ppl.finalize();
  (*input_dict)["top"] = top;
  (*input_dict)["bottom"] = bottom;
  (*input_dict)["left"] = left;
  (*input_dict)["right"] = right;
  (*input_dict)["ratio"] = 1 / ratio;
}

IPIPE_REGISTER(Backend, PPLResizeCenterPadTensor, "PPLResizeCenterPadTensor")
}  // namespace ipipe

namespace ipipe {
bool PPLResizePadTensor::init(const std::unordered_map<std::string, std::string>& config,
                              dict dict_config) {
  params_ = std::unique_ptr<Params>(new Params({{"max_h", "0"}, {"max_w", "0"}}, {}, {}, {}));
  if (!params_->init(config)) return false;

  TRACE_EXCEPTION(max_h_ = std::stoi(params_->operator[]("max_h")));
  TRACE_EXCEPTION(max_w_ = std::stoi(params_->operator[]("max_w")));
  IPIPE_ASSERT(max_h_ > 0 && max_w_ > 0 && max_h_ * max_w_ < 1024 * 1024);

  return true;
}

void PPLResizePadTensor::forward(dict input_dict) {
  at::Tensor in_tensor = dict_get<at::Tensor>(input_dict, TASK_DATA_KEY);
  in_tensor = img_hwc_guard(in_tensor);
  int img_h = in_tensor.sizes()[0];
  int img_w = in_tensor.sizes()[1];
  int c = in_tensor.sizes()[2];
  IPIPE_ASSERT(c == 3);

  float ratio = float(img_w) / float(img_h);
  int resize_h;
  int resize_w;
  if (max_h_ * ratio <= max_w_) {
    resize_w = std::min(int(std::round(max_h_ * ratio)), max_w_);
    resize_h = max_h_;
  } else {
    resize_w = max_w_;
    resize_h = std::min(int(std::round(max_w_ / ratio)), max_h_);
  }

  HWCTensorWrapper ppl(input_dict, max_h_, max_w_, true);
  auto& input_tensor = ppl.input_tensor;
  auto& output_tensor = ppl.output_tensor;

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  if (input_tensor.scalar_type() == at::kByte) {
    unsigned char* image = input_tensor.data_ptr<unsigned char>();
    unsigned char* output = output_tensor.data_ptr<unsigned char>();

    assert(max_w_ * c == output_tensor.stride(0));
    assert(img_w * c == input_tensor.stride(0));
    auto ret = ppl::cv::cuda::Resize<unsigned char, 3>(
        stream, img_h, img_w, input_tensor.stride(0), image, resize_h, resize_w,
        output_tensor.stride(0), output, ppl::cv::INTERPOLATION_LINEAR);
    if (ret != ppl::common::RC_SUCCESS) {
      SPDLOG_ERROR("PPLResizePadTensor: Resize failed, ret = {}", ret);
      return;
    }
  } else if (input_tensor.scalar_type() == at::kFloat) {
    float* image = input_tensor.data_ptr<float>();
    float* output = output_tensor.data_ptr<float>();

    auto ret = ppl::cv::cuda::Resize<float, 3>(stream, img_h, img_w, input_tensor.stride(0), image,
                                               resize_h, resize_w, output_tensor.stride(0), output,
                                               ppl::cv::INTERPOLATION_LINEAR);
    if (ret != ppl::common::RC_SUCCESS) {
      SPDLOG_ERROR("PPLResizePadTensor: Resize failed");
      return;
    }
  } else {
    throw std::invalid_argument("Only support type of float or char.");
  }
  ppl.finalize();
  (*input_dict)["ratio_h"] = static_cast<float>(img_h) / resize_h;
  (*input_dict)["ratio_w"] = static_cast<float>(img_w) / resize_w;

  float x_ratio = img_w * 1.0f / resize_w;
  float y_ratio = img_h * 1.0f / resize_h;
  std::function<std::pair<float, float>(float, float)> inverse_trans = [x_ratio, y_ratio](float x,
                                                                                          float y) {
    return std::pair<float, float>(x_ratio * x, y_ratio * y);
  };
  (*input_dict)["inverse_trans"] = inverse_trans;
}

IPIPE_REGISTER(Backend, PPLResizePadTensor, "PPLResizePadTensor")
}  // namespace ipipe