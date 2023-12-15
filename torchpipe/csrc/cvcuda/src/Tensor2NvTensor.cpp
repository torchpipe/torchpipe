#include "Tensor2NvTensor.hpp"
#include "reflect.h"
#include <ATen/ATen.h>
#include <nvcv/Tensor.hpp>
#include <nvcv/DataType.hpp>

#include "torch_utils.hpp"

namespace {

nvcv::ImageFormat torch2nvcv(at::ScalarType dtype) {
  switch (dtype) {
    case at::kByte:
      return nvcv::FMT_RGB8;
    case at::kFloat:
      return nvcv::FMT_RGBf32;
    case at::kHalf:
      return nvcv::FMT_RGBf16;
    default:
      throw std::runtime_error("torch2nvcv: Unsupported dtype");
  }
}

struct ATenNvcvTensor {
  at::Tensor handle;
  //   nvcv::Tensor tensor;
};

void deleter(void *ctx, const NVCVTensorData *data) { delete static_cast<ATenNvcvTensor *>(ctx); }

nvcv::Tensor torchTensortoNvcvTensor(at::Tensor data) {
  // todo FillNVCVTensorData
  nvcv::Tensor::Requirements inReqs;
  std::unique_ptr<ATenNvcvTensor> ctx(new ATenNvcvTensor());
  NVCVTensorHandle handle;
  if (ipipe::is_contiguous_wrt_nchw(data)) {
    // todo
    data = ipipe::img_hwc_guard(data);
    data = data.contiguous();
    return torchTensortoNvcvTensor(data);
    // data = ipipe::img_nchw_guard(data);
    // ctx->handle = data;

    // int maxImageWidth = data.size(3);
    // int maxImageHeight = data.size(2);
    // int c = data.size(1);
    // int batch = data.size(0);

    // CalcRequirements(int rank, const int64_t *shape, const DataType &dtype, NVCVTensorLayout
    // layout,
    //                  int32_t baseAlign, int32_t rowAlign);

    // inReqs = nvcv::Tensor::CalcRequirements(batch, {c, maxImageWidth, maxImageHeight},
    //                                         torch2nvcv(data.scalar_type()));
  } else if (ipipe::is_contiguous_wrt_hwc(data)) {
    data = ipipe::img_hwc_guard(data);
    ctx->handle = data;

    int maxImageWidth = data.size(1);
    int maxImageHeight = data.size(0);
    int c = data.size(2);
    int batch = 1;

    nvcv::TensorDataStridedCuda::Buffer inBuf;
    inBuf.strides[3] = data.element_size();
    inBuf.strides[2] = c * inBuf.strides[3];
    inBuf.strides[1] = maxImageWidth * inBuf.strides[2];
    inBuf.strides[0] = maxImageHeight * inBuf.strides[1];
    inBuf.basePtr = reinterpret_cast<NVCVByte *>(data.data_ptr());
    inReqs = nvcv::Tensor::CalcRequirements(batch, {maxImageWidth, maxImageHeight},
                                            torch2nvcv(data.scalar_type()));
    nvcv::TensorDataStridedCuda inData(nvcv::TensorShape{inReqs.shape, inReqs.rank, inReqs.layout},
                                       nvcv::DataType{inReqs.dtype}, inBuf);

    nvcv::detail::CheckThrow(nvcvTensorWrapDataConstruct(
        &inData.cdata(), deleter, static_cast<void *>(ctx.release()), &handle));
  } else if (ipipe::is_nchw(data) || ipipe::is_hwc(data)) {
    data = data.contiguous();
    return torchTensortoNvcvTensor(data);
  } else {
    throw std::runtime_error("torchTensortoNvcvTensor: Unsupported layout");
  }

  return nvcv::Tensor(std::move(handle));
}
}  // namespace
namespace ipipe {

bool Tensor2NvTensor::init(const std::unordered_map<std::string, std::string> &, dict) {
  return true;
}

void Tensor2NvTensor::forward(dict input_dict) {
  auto &input = *input_dict;

  auto data = dict_get<at::Tensor>(input_dict, TASK_DATA_KEY);
  auto result = torchTensortoNvcvTensor(data);  // true is for 'deepcopy'
  input[TASK_RESULT_KEY] = result;
}

IPIPE_REGISTER(Backend, Tensor2NvTensor, "Tensor2NvTensor");

}  // namespace ipipe