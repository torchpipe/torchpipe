
#include "PPLCopyMakeBorderTensor.hpp"

#include <ppl/cv/cuda/copymakeborder.h>
// #include <torch/torch.h>
#include "c10/cuda/CUDAStream.h"

#include <ATen/ATen.h>

#include "ipipe_utils.hpp"
#include "base_logging.hpp"
#include "reflect.h"
#include "exception.hpp"
#include "ppl_cv_helper.hpp"

namespace ipipe {
bool PPLCopyMakeBorderTensor::init(const std::unordered_map<std::string, std::string>& config,
                                   dict dict_config) {
  // params_ = std::unique_ptr<Params>(new Params({}, {"top", "bottom", "left", "right"}, {}, {}));
  // if (!params_->init(config)) return false;

  // TRACE_EXCEPTION(top_ = std::stoi(params_->operator[]("top")));
  // TRACE_EXCEPTION(bottom_ = std::stoi(params_->operator[]("bottom")));
  // TRACE_EXCEPTION(left_ = std::stoi(params_->operator[]("left")));
  // TRACE_EXCEPTION(right_ = std::stoi(params_->operator[]("right")));

  return true;
}
void PPLCopyMakeBorderTensor::forward(dict input_dict) {
  top_ = dict_get<int>(input_dict, "top");
  bottom_ = dict_get<int>(input_dict, "bottom");
  left_ = dict_get<int>(input_dict, "left");
  right_ = dict_get<int>(input_dict, "right");

  HWCTensorWrapper ppl(input_dict, top_, bottom_, left_, right_);
  auto& input_tensor = ppl.input_tensor;
  auto& output_tensor = ppl.output_tensor;

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  int img_h = input_tensor.sizes()[0];
  int img_w = input_tensor.sizes()[1];
  int c = input_tensor.sizes()[2];

  if (input_tensor.scalar_type() == at::kByte) {
    unsigned char* image = input_tensor.data_ptr<unsigned char>();
    unsigned char* output = output_tensor.data_ptr<unsigned char>();

    auto ret = ppl::cv::cuda::CopyMakeBorder<unsigned char, 3>(
        stream, img_h, img_w, input_tensor.stride(0), image, output_tensor.stride(0), output, top_,
        bottom_, left_, right_, ppl::cv::BORDER_CONSTANT, 0);
    if (ret != ppl::common::RC_SUCCESS) {
      SPDLOG_ERROR("PPLCopyMakeBorderTensor: Resize failed, ret = {}", ret);
      return;
    }

  } else if (input_tensor.scalar_type() == at::kFloat) {
    float* image = input_tensor.data_ptr<float>();
    float* output = output_tensor.data_ptr<float>();

    auto ret = ppl::cv::cuda::CopyMakeBorder<float, 3>(
        stream, img_h, img_w, input_tensor.stride(0), image, output_tensor.stride(0), output, top_,
        bottom_, left_, right_, ppl::cv::BORDER_CONSTANT, 0);
    if (ret != ppl::common::RC_SUCCESS) {
      SPDLOG_ERROR("PPLCopyMakeBorderTensor: Resize failed");
      return;
    }
  } else {
    throw std::invalid_argument("error datatype of tensor. Need datatype float or char.");
  }
  ppl.finalize();
}

IPIPE_REGISTER(Backend, PPLCopyMakeBorderTensor, "PPLCopyMakeBorderTensor")
}  // namespace ipipe