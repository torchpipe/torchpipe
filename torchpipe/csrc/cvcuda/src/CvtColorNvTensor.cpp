#include "CvtColorNvTensor.hpp"
#include "ipipe_common.hpp"
#include "exception.hpp"

#include "cvcuda_helper.hpp"
#include "c10/cuda/CUDAStream.h"

namespace ipipe {
bool CvtColorNvTensor::init(const std::unordered_map<std::string, std::string>& config, dict) {
  params_ = std::unique_ptr<Params>(new Params({}, {"color"}, {}, {}));
  if (!params_->init(config)) return false;

  TRACE_EXCEPTION(color_ = (params_->operator[]("color")));

  return true;
}

void CvtColorNvTensor::forward(dict input_dict) {
  std::string input_color;
  TRACE_EXCEPTION(input_color = any_cast<std::string>(input_dict->at("color")));
  if (input_color == color_) {
    (*input_dict)[TASK_RESULT_KEY] = (*input_dict)[TASK_DATA_KEY];
    return;
  }
  if (input_color != "rgb" && input_color != "bgr") {
    throw std::invalid_argument("input_color should be rgb or bgr, but is " + input_color);
  }

  nvcv::Tensor data = dict_get<nvcv::Tensor>(input_dict, TASK_DATA_KEY);
  IPIPE_ASSERT(data.layout() == nvcv::TENSOR_HWC);

  nvcv::Tensor resultTensor(data.shape(), data.dtype(), nvcv::MemAlignment{}.rowAddr(1).baseAddr(0),
                            nvcv_torch_allocator());

  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  cvtColorOp_(stream, data, resultTensor, NVCV_COLOR_BGR2RGB);

  (*input_dict)["color"] = color_;
  (*input_dict)[TASK_RESULT_KEY] = resultTensor;
}
IPIPE_REGISTER(Backend, CvtColorNvTensor, "CvtColorNvTensor,cvtColorNvTensor");

}  // namespace ipipe