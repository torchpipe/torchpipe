#include "helper/torch.hpp"

#include "nvjpeg_torch/DecodeTensor.hpp"
#include <c10/cuda/CUDAStream.h>
#include <torch/torch.h>
namespace str = omniback::str;
namespace {
[[maybe_unused]] void check_nvjpeg_result(nvjpegStatus_t _e) {
  OMNI_ASSERT(
      _e == NVJPEG_STATUS_SUCCESS,
      ("nvjpeg error: nvjpegStatus_t = " + std::to_string(int(_e))));
}

#define CHECK_NVJPEG_RESULT(EVAL)                               \
  {                                                             \
    nvjpegStatus_t _e = EVAL;                                   \
    if (_e != NVJPEG_STATUS_SUCCESS) {                          \
      throw std::runtime_error(                                 \
          str::format(                                          \
              "nvjpeg error: nvjpegStatus_t = {0}, EVAL = {1}", \
              int(_e),                                          \
              #EVAL));                                          \
    }                                                           \
  }

bool decode(
    const std::string& data,
    nvjpegHandle_t handle,
    nvjpegJpegState_t state,
    torch::Tensor& image_tensor,
    const std::string& color,
    const std::string& data_format) {
  const auto* blob = (const unsigned char*)data.data();
  int nComponents;
  nvjpegChromaSubsampling_t subsampling;
  int widths[NVJPEG_MAX_COMPONENT];
  int heights[NVJPEG_MAX_COMPONENT];

  auto re = nvjpegGetImageInfo(
      handle, blob, data.length(), &nComponents, &subsampling, widths, heights);
  if (NVJPEG_STATUS_SUCCESS != re) {
    SPDLOG_WARN(
        "nvjpegGetImageInfo failed. data.length() = {} result = {}",
        data.length(),
        int(re));
    return false;
  }

  // if (nComponents == 1)
  // {

  //     SPDLOG_WARN(
  //         "Forcing conversion from nComponents == 1 to nComponents == 3");
  //     // nComponents = 3;
  // }
  if (nComponents != 3) {
    SPDLOG_ERROR(
        "Invalid channel count: {}. Supported values: 3. (This is might be "
        "recoverable - falling back to CPU decoding)",
        nComponents);
    return false;
  }
  //   if (!SupportedSubsampling(subsampling)) {
  //     SPDLOG_ERROR("subsampling not supported");
  //     return false;
  //   }

  int h = heights[0];
  int w = widths[0];

  // size_t image_size = h * w * nComponents;

  auto options = torch::TensorOptions()
                     .device(torch::kCUDA, -1)
                     .dtype(torch::kByte)
                     .layout(torch::kStrided)
                     .requires_grad(false);
  if (data_format == "nchw") {
    image_tensor = torch::empty(
        {1, nComponents, h, w}, options, torch::MemoryFormat::Contiguous);
  } else {
    image_tensor = torch::empty(
        {h, w, nComponents}, options, torch::MemoryFormat::Contiguous);
  }
  auto* image = image_tensor.data_ptr<unsigned char>();

  decltype(NVJPEG_OUTPUT_BGR) target_color;

  if (color == "rgb") {
    if (data_format == "nchw")
      target_color = NVJPEG_OUTPUT_RGB;
    else
      target_color = NVJPEG_OUTPUT_RGBI;
  } else {
    if (data_format == "nchw")
      target_color = NVJPEG_OUTPUT_BGR;
    else
      target_color = NVJPEG_OUTPUT_BGRI;
  }
  nvjpegImage_t nv_image;
  if (data_format == "nchw") {
    for (size_t i = nComponents; i < NVJPEG_MAX_COMPONENT; i++) {
      nv_image.channel[i] = nullptr;
      nv_image.pitch[i] = 0;
    }
    for (size_t i = 0; i < nComponents; i++) {
      nv_image.channel[i] = image + i * w * h;
      nv_image.pitch[i] = w;
    }
  } else {
    for (size_t i = 1; i < NVJPEG_MAX_COMPONENT; i++) {
      nv_image.channel[i] = nullptr;
      nv_image.pitch[i] = 0;
    }
    nv_image.channel[0] = image;
    nv_image.pitch[0] = w * nComponents;
  }

  if (NVJPEG_STATUS_SUCCESS !=
      nvjpegDecode(
          handle,
          state,
          blob,
          data.length(),
          target_color,
          &nv_image,
          c10::cuda::getCurrentCUDAStream())) {
    SPDLOG_WARN("nvjpegDecode failed");
    return false;
  }

  return true;
}
} // namespace

namespace torchpipe {

DecodeTensor::~DecodeTensor() {
  nvjpegJpegStateDestroy(state_);
  nvjpegDestroy(handle_);
}
void DecodeTensor::impl_init(
    const std::unordered_map<std::string, std::string>& config,
    const omniback::dict& kwargs) {
  str::try_update(config, "color", color_);
  str::try_update(config, "data_format", data_format_);

  OMNI_ASSERT(color_ == "rgb" || color_ == "bgr");
  OMNI_ASSERT(data_format_ == "nchw" || data_format_ == "hwc");

  auto tmp =
      torch::empty({1, 1}, torch::TensorOptions().device(torch::kCUDA, -1));
  nvjpegBackend_t backend = NVJPEG_BACKEND_DEFAULT;
  // backend = NVJPEG_BACKEND_GPU_HYBRID;
//
#if NVJPEG_VER_MAJOR == 11 && NVJPEG_VER_MINOR <= 6
  // nvjpegDevAllocator_t dev_allocator = {&torch_malloc, &torch_free};
  CHECK_NVJPEG_RESULT(nvjpegCreateEx(backend, nullptr, nullptr, 0, &handle_));
#else
  nvjpegDevAllocatorV2_t dev_allocator_async = {
      &torch_malloc_async, &torch_free_async};
  nvjpegPinnedAllocatorV2_t pinned_allocator = {
      &torch_pinned_malloc_async, &torch_pinned_free_async};
  // todo nvjpegDevAllocatorV2_t nvjpegCreateExV2
  CHECK_NVJPEG_RESULT(nvjpegCreateExV2(
      backend, &dev_allocator_async, &pinned_allocator, 0, &handle_));
#endif

  CHECK_NVJPEG_RESULT(nvjpegJpegStateCreate(handle_, &state_));

  //   constexpr int device_padding = 256;
  //   constexpr int host_padding = 256;
  //   check_nvjpeg_result(nvjpegSetDeviceMemoryPadding(device_padding,
  //   handle_));
  //   check_nvjpeg_result(nvjpegSetPinnedMemoryPadding(host_padding,
  //   handle_));
}

void DecodeTensor::forward(const omniback::dict& input_dict) {
  auto& input = *input_dict;

  std::string data = input[TASK_DATA_KEY].cast<std::string>();

  at::Tensor tensor;

  if (decode(data, handle_, state_, tensor, color_, data_format_)) {
    input[TASK_RESULT_KEY] = tensor;
    input["color"] = color_;
    input["data_format"] = data_format_;
  } else {
    SPDLOG_WARN("gpu decode failed");
  }
}

OMNI_REGISTER(omniback::Backend, DecodeTensor, "DecodeTensor");
} // namespace torchpipe