
#include "PPLWarpAffineTensor.hpp"

#include <ppl/cv/cuda/warpaffine.h>
// #include <torch/torch.h>
#include "c10/cuda/CUDAStream.h"

#include <ATen/ATen.h>

#include "ipipe_utils.hpp"
#include "base_logging.hpp"
#include "reflect.h"
#include "exception.hpp"
#include "ppl_cv_helper.hpp"

namespace ipipe {
bool PPLWarpAffineTensor::init(const std::unordered_map<std::string, std::string>& config,
                               dict dict_config) {
  params_ = std::unique_ptr<Params>(new Params({{"target_h", "0"}, {"target_w", "0"}}, {}, {}, {}));
  if (!params_->init(config)) return false;

  TRACE_EXCEPTION(target_h_ = std::stoi(params_->operator[]("target_h")));
  TRACE_EXCEPTION(target_w_ = std::stoi(params_->operator[]("target_w")));
  // if (target_h_ > 1024 * 1024 || target_w_ > 1024 * 1024 || target_h_ < 1 || target_w_ < 1) {
  //   SPDLOG_ERROR("PPLWarpAffineTensor: illigle h or w: h=" + std::to_string(target_h_) +
  //                "w=" + std::to_string(target_w_));
  //   return false;
  // }

  return true;
}
void PPLWarpAffineTensor::forward(dict input_dict) {
  try_update(input_dict, "target_h", target_h_);
  try_update(input_dict, "target_w", target_w_);
  IPIPE_ASSERT(target_h_ > 0 && target_w_ > 0 && target_h_ * target_w_ < 1024 * 1024);

  auto affine_matrix = dict_get<std::vector<float>>(input_dict, "affine_matrix");

  if (affine_matrix.size() != 6) {
    SPDLOG_ERROR("PPLWarpAffineTensor: affine_matrix.size() != 6");
    return;
  }

  HWCTensorWrapper ppl(input_dict, target_h_, target_w_);
  auto& input_tensor = ppl.input_tensor;
  auto& output_tensor = ppl.output_tensor;

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  int img_h = input_tensor.sizes()[0];
  int img_w = input_tensor.sizes()[1];
  int c = input_tensor.sizes()[2];

  if (input_tensor.scalar_type() == at::kByte) {
    unsigned char* image = input_tensor.data_ptr<unsigned char>();
    unsigned char* output = output_tensor.data_ptr<unsigned char>();
    assert(target_w_ * c == output_tensor.stride(0));
    assert(img_w * c == input_tensor.stride(0));
    auto ret = ppl::cv::cuda::WarpAffine<unsigned char, 3>(
        stream, img_h, img_w, input_tensor.stride(0), image, target_h_, target_w_,
        output_tensor.stride(0), output, affine_matrix.data(), ppl::cv::INTERPOLATION_LINEAR,
        ppl::cv::BORDER_CONSTANT, (unsigned char)0);
    if (ret != ppl::common::RC_SUCCESS) {
      SPDLOG_ERROR("PPLWarpAffineTensor: WarpAffine failed, ret = {}", ret);
      return;
    }

  } else if (input_tensor.scalar_type() == at::kFloat) {
    float* image = input_tensor.data_ptr<float>();
    float* output = output_tensor.data_ptr<float>();
    assert(output_tensor.stride(0) == target_w_ * c);
    auto ret = ppl::cv::cuda::WarpAffine<float, 3>(
        stream, img_h, img_w, input_tensor.stride(0), image, target_h_, target_w_,
        output_tensor.stride(0), output, affine_matrix.data(), ppl::cv::INTERPOLATION_LINEAR,
        ppl::cv::BORDER_CONSTANT, (float)0);
    if (ret != ppl::common::RC_SUCCESS) {
      SPDLOG_ERROR("PPLWarpAffineTensor: WarpAffine failed");
      return;
    }
  } else {
    throw std::invalid_argument("error datatype of tensor.  need datatype float or char.");
  }
  ppl.finalize();
}

IPIPE_REGISTER(Backend, PPLWarpAffineTensor, "PPLWarpAffineTensor")
}  // namespace ipipe