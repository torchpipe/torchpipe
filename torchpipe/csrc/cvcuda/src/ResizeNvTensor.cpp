#include "ResizeNvTensor.hpp"
#include "ipipe_common.hpp"
#include "exception.hpp"

#include "cvcuda_helper.hpp"
#include "c10/cuda/CUDAStream.h"

#include "base_logging.hpp"

namespace ipipe {
bool ResizeNvTensor::init(const std::unordered_map<std::string, std::string> &config, dict) {
  params_ = std::unique_ptr<Params>(new Params({}, {"resize_h", "resize_w"}, {}, {}));
  if (!params_->init(config)) return false;

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

void ResizeNvTensor::forward(dict input_dict) {
  nvcv::Tensor data = dict_get<nvcv::Tensor>(input_dict, TASK_DATA_KEY);
  IPIPE_ASSERT(data.layout() == nvcv::TENSOR_HWC);

  const nvcv::TensorShape &tshape = data.shape();

  nvcv::TensorShape shapeHWC({resize_h_, resize_w_, tshape[2]}, "HWC");

  nvcv::Tensor resultTensor(shapeHWC, data.dtype(), nvcv::MemAlignment{}.rowAddr(1).baseAddr(0),
                            nvcv_torch_allocator());

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  ResizeOp_(stream, data, resultTensor, NVCV_INTERP_LINEAR);

  (*input_dict)[TASK_RESULT_KEY] = resultTensor;
}
IPIPE_REGISTER(Backend, ResizeNvTensor, "ResizeNvTensor");

}  // namespace ipipe