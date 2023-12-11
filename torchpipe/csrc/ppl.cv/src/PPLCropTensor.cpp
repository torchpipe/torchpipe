
#include "PPLCropTensor.hpp"

#include <ppl/cv/cuda/crop.h>
// #include <torch/torch.h>
#include "c10/cuda/CUDAStream.h"

#include <ATen/ATen.h>

#include "ipipe_utils.hpp"
#include "base_logging.hpp"
#include "reflect.h"
#include "exception.hpp"
#include "ppl_cv_helper.hpp"

namespace ipipe {
bool PPLCropTensor::init(const std::unordered_map<std::string, std::string>& config,
                         dict dict_config) {
  params_ = std::unique_ptr<Params>(new Params({{"target_h", "0"}, {"target_w", "0"}}, {}, {}, {}));
  if (!params_->init(config)) return false;

  // TRACE_EXCEPTION(target_h = std::stoi(params_->operator[]("target_h")));
  // TRACE_EXCEPTION(target_w = std::stoi(params_->operator[]("target_w")));
  return true;
}

void PPLCropTensor::forward(dict input_dict) {
  std::vector<int> box;
  update(input_dict, TASK_BOX_KEY, box);
  IPIPE_ASSERT(box.size() == 4 && box[0] > 0 && box[1] > 0 && box[2] > box[0] && box[3] > box[1]);
  float scale = 1;
  int x1 = box[0];
  int y1 = box[1];
  int target_h = box[3] - box[1];
  int target_w = box[2] - box[0];
  HWCTensorWrapper ppl(input_dict, target_h, target_w);
  auto& input_tensor = ppl.input_tensor;
  auto& output_tensor = ppl.output_tensor;

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  int img_h = input_tensor.sizes()[0];
  int img_w = input_tensor.sizes()[1];
  int c = input_tensor.sizes()[2];

  if (input_tensor.scalar_type() == at::kByte) {
    unsigned char* image = input_tensor.data_ptr<unsigned char>();
    unsigned char* output = output_tensor.data_ptr<unsigned char>();
    assert(target_w * c == output_tensor.stride(0));
    assert(img_w * c == input_tensor.stride(0));
    auto ret = ppl::cv::cuda::Crop<unsigned char, 3>(
        stream, img_h, img_w, input_tensor.stride(0), image, target_h, target_w,
        output_tensor.stride(0), output, x1, y1, scale);
    if (ret != ppl::common::RC_SUCCESS) {
      SPDLOG_ERROR("PPLCropTensor: Resize failed, ret = {}", ret);
      return;
    }

  } else if (input_tensor.scalar_type() == at::kFloat) {
    float* image = input_tensor.data_ptr<float>();
    float* output = output_tensor.data_ptr<float>();

    auto ret =
        ppl::cv::cuda::Crop<float, 3>(stream, img_h, img_w, input_tensor.stride(0), image, target_h,
                                      target_w, output_tensor.stride(0), output, x1, y1, scale);
    if (ret != ppl::common::RC_SUCCESS) {
      SPDLOG_ERROR("PPLCropTensor: Resize failed");
      return;
    }
  } else {
    throw std::invalid_argument("error datatype of tensor.  need datatype float or char.");
  }
  ppl.finalize();
}

IPIPE_REGISTER(Backend, PPLCropTensor, "PPLCropTensor")
}  // namespace ipipe