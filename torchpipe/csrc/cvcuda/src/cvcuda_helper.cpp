
#include "cvcuda_helper.hpp"
#include <nvcv/Tensor.hpp>
#include <nvcv/DataType.hpp>
#include <nvcv/TensorData.h>
#include "torch_utils.hpp"
#include <ATen/DLConvertor.h>
#include <ATen/dlpack.h>
#include <nvcv/alloc/Allocator.hpp>
#include <c10/cuda/CUDACachingAllocator.h>
#include "base_logging.hpp"
#include <optional>

#ifndef DLPACK_VERSION
#ifdef DLPACK_MAJOR_VERSION
#define DLPACK_VERSION (DLPACK_MAJOR_VERSION * 100 + DLPACK_MINOR_VERSION * 1)
#else
#define DLPACK_VERSION 0
#endif
#endif

#include <memory>
namespace ipipe {
namespace {

nvcv::DataType ToNVCVDataType(const DLDataType& dtype) {
  nvcv::PackingParams pp;
  pp.byteOrder = nvcv::ByteOrder::MSB;

  int lanes = dtype.lanes;
  int bits = dtype.bits;

  switch (lanes) {
    case 1:
      pp.swizzle = nvcv::Swizzle::S_X000;
      break;
    case 2:
      pp.swizzle = nvcv::Swizzle::S_XY00;
      break;
    case 3:
      pp.swizzle = nvcv::Swizzle::S_XYZ0;
      break;
    case 4:
      pp.swizzle = nvcv::Swizzle::S_XYZW;
      break;
    default:
      throw std::runtime_error("DLPack buffer's data type must have at most 4 lanes");
  }

  for (int i = 0; i < lanes; ++i) {
    pp.bits[i] = bits;
  }
  for (int i = lanes; i < 4; ++i) {
    pp.bits[i] = 0;
  }

  nvcv::Packing packing = nvcv::MakePacking(pp);

  nvcv::DataKind kind;

  switch (dtype.code) {
    // case kDLBool:
    case kDLInt:
      kind = nvcv::DataKind::SIGNED;
      break;
    case kDLUInt:
      kind = nvcv::DataKind::UNSIGNED;
      break;
    case kDLComplex:
      kind = nvcv::DataKind::COMPLEX;
      break;
    case kDLFloat:
      kind = nvcv::DataKind::FLOAT;
      break;
    default:
      throw std::runtime_error("Data type code not supported, must be Int, UInt, Float, Complex");
  }

  return nvcv::DataType(kind, packing);
}

DLDataType ToDLDataType(const nvcv::DataType& dataType) {
  DLDataType dt = {};
  dt.lanes = dataType.numChannels();

  switch (dataType.dataKind()) {
    case nvcv::DataKind::UNSIGNED:
      dt.code = kDLUInt;
      break;
    case nvcv::DataKind::SIGNED:
      dt.code = kDLInt;
      break;
    case nvcv::DataKind::FLOAT:
      dt.code = kDLFloat;
      break;
    case nvcv::DataKind::COMPLEX:
      dt.code = kDLComplex;
      break;
    default:
      throw std::runtime_error(
          "Data kind not supported, must be UNSIGNED, SIGNED, FLOAT or COMPLEX");
  }

  std::array<int32_t, 4> bpc = dataType.bitsPerChannel();

  for (int i = 1; i < dataType.numChannels(); ++i) {
    if (bpc[i] != bpc[0]) {
      throw std::runtime_error("All lanes must have the same bit depth");
    }
  }

  dt.bits = bpc[0];

  return dt;
}

NVCVTensorData FillNVCVTensorData(const DLTensor& tensor,
                                  std::optional<nvcv::TensorLayout> layout) {
  NVCVTensorData tensorData = {};

  // dtype ------------
  tensorData.dtype = ToNVCVDataType(tensor.dtype);

  // layout ------------
  if (layout) {
    tensorData.layout = *layout;
  }

  // rank ------------
  {
    // TODO: Add 0D support
    int rank = tensor.ndim == 0 ? 1 : tensor.ndim;
    if (rank < 1 || rank > NVCV_TENSOR_MAX_RANK) {
      throw std::invalid_argument("Error Number of dimensions: " + std::to_string(rank) +
                                  " is not supported");
    }
    tensorData.rank = rank;
  }

  // shape ------------
  std::copy_n(tensor.shape, tensor.ndim, tensorData.shape);

#if DLPACK_VERSION > 40
  // buffer type ------------
  if ((kDLCUDAHost == tensor.device.device_type || kDLCUDA == tensor.device.device_type ||
       kDLCUDAManaged == tensor.device.device_type)) {
#else
  if ((kDLGPU == tensor.device.device_type)) {
#endif
    tensorData.bufferType = NVCV_TENSOR_BUFFER_STRIDED_CUDA;
  } else {
    throw std::runtime_error("Only CUDA-accessible tensors are supported for now");
  }

  NVCVTensorBufferStrided& dataStrided = tensorData.buffer.strided;

  // stride ------------
  int elemStrideBytes = (tensor.dtype.bits * tensor.dtype.lanes + 7) / 8;
  for (int d = 0; d < tensor.ndim; ++d) {
    dataStrided.strides[d] = tensor.strides[d] * elemStrideBytes;
  }

  // Memory buffer ------------
  dataStrided.basePtr = reinterpret_cast<NVCVByte*>(tensor.data) + tensor.byte_offset;

  return tensorData;
}
void DLManagedTensor_deleter(void* ctx, const NVCVTensorData* data) {
  DLManagedTensor* p = static_cast<DLManagedTensor*>(ctx);
  p->deleter(p);
}

}  // namespace
nvcv::Tensor toNvcvTensor(const torch::Tensor& src, std::string data_format) {
  DLManagedTensor* dlMTensor = torch::toDLPack(src);

  std::optional<nvcv::TensorLayout> layout;
  if (data_format == "hwc") {
    layout = nvcv::TENSOR_HWC;
  } else if (data_format == "chw") {
    layout = nvcv::TENSOR_CHW;
  } else if (data_format == "nchw") {
    layout = nvcv::TENSOR_NCHW;
  } else if (data_format == "nhwc") {
    layout = nvcv::TENSOR_NHWC;
  } else if (data_format == "") {
    layout = std::nullopt;
  } else {
    throw std::invalid_argument("Unsupported data format: " + data_format);
  }
  NVCVTensorData data = FillNVCVTensorData(dlMTensor->dl_tensor, layout);
  NVCVTensorHandle handle;
  nvcv::detail::CheckThrow(nvcvTensorWrapDataConstruct(&data, DLManagedTensor_deleter,
                                                       static_cast<void*>(dlMTensor), &handle));

  return nvcv::Tensor(std::move(handle));
}

struct NvcvDLMTensor {
  nvcv::Tensor handle;
  DLManagedTensor tensor;
};

torch::Tensor fromNvcvTensor(const nvcv::Tensor& src) {
  NvcvDLMTensor* nvcvDLMTensor(new NvcvDLMTensor);
  nvcvDLMTensor->handle = src;

  nvcvDLMTensor->tensor.manager_ctx = nvcvDLMTensor;

  DLManagedTensor& src_tensor = nvcvDLMTensor->tensor;

  src_tensor.deleter = [](DLManagedTensor* self) {
    delete[] self->dl_tensor.shape;
    delete[] self->dl_tensor.strides;
    delete static_cast<NvcvDLMTensor*>(self->manager_ctx);
  };

  try {
    auto p_stride = src.exportData<nvcv::TensorDataStrided>();

    if (!p_stride) {
      throw std::runtime_error("Only strided tensors are supported for now");
    }
    nvcv::TensorDataStrided& tensorData = *p_stride;
    DLTensor& tensor = src_tensor.dl_tensor;

    // Set up device
    if (tensorData.IsCompatible<nvcv::TensorDataStridedCuda>()) {
// TODO: detect correct device_type from memory buffer
#if DLPACK_VERSION > 40
      tensor.device.device_type = kDLCUDA;
#else
      tensor.device.device_type = kDLGPU;
#endif
      // TODO: detect correct device_id from memory buffer (if possible)
      tensor.device.device_id = 0;
    } else {
      throw std::runtime_error(
          "Tensor buffer type not supported, must be either CUDA or Host (CPU)");
    }

    // Set up ndim
    tensor.ndim = tensorData.rank();

    // Set up data
    tensor.data = tensorData.basePtr();
    tensor.byte_offset = 0;

    // Set up shape
    tensor.shape = new int64_t[tensor.ndim];
    std::copy_n(tensorData.shape().shape().begin(), tensor.ndim, tensor.shape);

    // Set up dtype
    tensor.dtype = ToDLDataType(tensorData.dtype());

    // Set up strides
    tensor.strides = new int64_t[tensor.ndim];
    for (int i = 0; i < tensor.ndim; ++i) {
      int64_t stride = tensorData.cdata().buffer.strided.strides[i];
      if (stride % tensorData.dtype().strideBytes() != 0) {
        throw std::runtime_error("Stride must be a multiple of the element size in bytes");
      }

      tensor.strides[i] =
          tensorData.cdata().buffer.strided.strides[i] / tensorData.dtype().strideBytes();
    }
  } catch (...) {
    src_tensor.deleter(&src_tensor);
    throw;
  }
  return torch::fromDLPack(&src_tensor);
}
nvcv::Allocator nvcv_torch_allocator() {
  nvcv::CustomAllocator myAlloc{nvcv::CustomCudaMemAllocator{
      [](int64_t size, int32_t bufAlign) {
        void* ptr = nullptr;
        // cudaMalloc(&ptr, size);

        ptr = torch::cuda::CUDACachingAllocator::raw_alloc(size);
        if (reinterpret_cast<ptrdiff_t>(ptr) % bufAlign) {
          SPDLOG_WARN("The torch allocator cannot align the address to the alignment {}", bufAlign);
        }
        return ptr;
      },
      [](void* ptr, int64_t bufLen, int32_t bufAlign) {
        // cudaFree(ptr);
        torch::cuda::CUDACachingAllocator::raw_delete(ptr);
      }}};
  return myAlloc;
}

}  // namespace ipipe