#include <ATen/ATen.h>
#include <nvcv/Tensor.hpp>
#include <nvcv/DataType.hpp>
#include <nvcv/TensorData.h>
#include "torch_utils.hpp"
namespace ipipe {

namespace cvcuda {

NVCVDataType torchtype2nvcv(at::ScalarType dtype) {
  switch (dtype) {
    case at::kFloat:
      return NVCV_DATA_TYPE_3F32;
    case at::kByte:
      return NVCV_DATA_TYPE_3U8;
    case at::kHalf:
      return NVCV_DATA_TYPE_3F16;
    case at::kInt:
      return NVCV_DATA_TYPE_3S32;
    case at::kLong:
      return NVCV_DATA_TYPE_3S64;
    default:
      throw std::runtime_error("torch2nvcv: Unsupported dtype");
  }
}

struct ATenNvcvTensor {
  at::Tensor handle;
  //   nvcv::Tensor tensor;
};

void deleter(void *ctx, const NVCVTensorData *data) { delete static_cast<ATenNvcvTensor *>(ctx); }
}  // namespace cvcuda
nvcv::Tensor torch2nvcv(at::Tensor data) {
  data = img_hwc_guard(data);
  if (!data.is_contiguous()) {
    data = data.contiguous();  // todo: unnecessary?
  }
  int h = data.size(0);
  int w = data.size(1);
  int c = data.size(2);

  NVCVTensorBufferStrided inBuf;
  inBuf.strides[2] = data.element_size();
  inBuf.strides[1] = c * inBuf.strides[2];
  inBuf.strides[0] = w * inBuf.strides[1];
  inBuf.basePtr = reinterpret_cast<NVCVByte *>(data.data_ptr());

  std::unique_ptr<cvcuda::ATenNvcvTensor> ctx(new cvcuda::ATenNvcvTensor());
  ctx->handle = data;

  NVCVTensorData inData;
  inData.dtype = ipipe::cvcuda::torchtype2nvcv(data.scalar_type());
  inData.layout = NVCV_TENSOR_HWC;
  inData.rank = 3;
  std::memcpy(inData.shape, data.sizes().data(), sizeof(int64_t) * 3);

  inData.bufferType = NVCV_TENSOR_BUFFER_STRIDED_CUDA;
  inData.buffer.strided = inBuf;

  NVCVTensorHandle handle;
  nvcv::detail::CheckThrow(nvcvTensorWrapDataConstruct(
      &inData, ipipe::cvcuda::deleter, static_cast<void *>(ctx.release()), &handle));
  // inData is copied indeed
  return nvcv::Tensor(std::move(handle));
}
}  // namespace ipipe