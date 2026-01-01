// Copyright 2021-2025 NetEase.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <atomic>
#include <chrono>
#include <mutex>
#include <thread>
#if 1
#include "c10/cuda/CUDAStream.h"
#endif
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAEvent.h>

// #include "time_utils.hpp"
// #include "base_logging.hpp"
// #include <torch/serialize.h>
#include <torch/extension.h>
#include <torch/torch.h>
#include <fstream>

#include <omniback/extension.hpp>
#include "helper/torch.hpp"
// #include "NvInferRuntime.h"
#include "omniback/helper/timer.hpp"
#include "helper/dlpack_helper.hpp"
#include <c10/cuda/CUDAStream.h>

namespace torchpipe {

void save(std::string save_name, torch::Tensor input) {
  std::vector<char> data_for_save = torch::pickle_save(input);
  // save_name
  std::ofstream fout(save_name, std::ios::out | std::ios::binary);
  fout.write(data_for_save.data(), data_for_save.size());
  fout.close();
}

void copy2ptr(torch::Tensor input, char* ptr) {
  IPIPE_ASSERT(
      input.is_contiguous(), "copy2ptr: input tensor must be contiguous");
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  size_t size = input.numel() * input.element_size();
  IPIPE_ASSERT(
      cudaMemcpyAsync(
          ptr, input.data_ptr(), size, cudaMemcpyDeviceToDevice, stream) ==
          cudaSuccess,
      "copy2ptr: cudaMemcpyAsync failed");
}

torch::Tensor load_tensor(std::string save_name) {
  std::ifstream file(save_name.c_str());
  if (!file.good()) {
    SPDLOG_ERROR("LoadTensor:  `" + save_name + "` not exists.");
    throw std::invalid_argument(" `" + save_name + "` not exists.");
  }
  file.seekg(0, file.end);
  int length = file.tellg();
  file.seekg(0, file.beg);

  std::vector<char> data(length);
  file.read(data.data(), length);

  auto data_loaded = torch::pickle_load(data).toTensor();
  return data_loaded;
}

#if 1
bool torch_not_use_default_stream(bool high_prio) {
  if (c10::cuda::getCurrentCUDAStream() == c10::cuda::getDefaultCUDAStream()) {
    c10::cuda::setCurrentCUDAStream(
        c10::cuda::getStreamFromPool(
            high_prio,
            -1)); // Schedule保证了init和forward在同一个线程
    return true;
  }
  return false;
}

bool torch_not_use_default_stream(int device_id, bool high_prio) {
  if (c10::cuda::current_device() != device_id && device_id >= 0) {
    c10::cuda::set_device(device_id);
  }
  if (c10::cuda::getCurrentCUDAStream(device_id) ==
      c10::cuda::getDefaultCUDAStream(device_id)) {
    c10::cuda::setCurrentCUDAStream(
        c10::cuda::getStreamFromPool(
            high_prio, device_id)); // Schedule保证了init和forward在同一个线程
    return true;
  }
  return false;
}

bool torch_is_using_default_stream() {
  if (c10::cuda::getCurrentCUDAStream(-1) ==
      c10::cuda::getDefaultCUDAStream(-1)) {
    return true;
  }
  return false;
}

// https://discuss.pytorch.org/t/asynchronous-copy-in-c-when-input-has-been-destructed/186515
torch::Tensor to_current_device(torch::Tensor input) {
  if (input.device() == torch::kCPU)
    return input.cuda();
  if (input.device().index() == c10::cuda::current_device())
    return input;
  torch::TensorOptions options;
  // input.is_pinned()
  return input.to(
      torch::TensorOptions().device(torch::kCUDA, -1),
      false,
      false,
      input.suggest_memory_format()); // 这里为异步操作, pytorch 自身cache
                                      // pinned memory， 不怕析构
}
#endif

bool is_cpu_tensor(torch::Tensor input) {
#ifndef TORCH_VERSION_MAJOR
  IPIPE_ASSERT(0, "TORCH_VERSION_MAJOR not defined");
#endif
#if TORCH_VERSION_MAJOR == 1 && TORCH_VERSION_MINOR < 10
  return input.device().is_cpu();
#else
  return input.is_cpu();
#endif
}
// Check if the given data variable is of CPU type.
bool is_any_cpu(omniback::any data) {
  if (auto tensor_opt = data.try_cast<torch::Tensor>()) {
    torch::Tensor tensor = tensor_opt.value();
    return is_cpu_tensor(tensor);
  }

  if (auto opt = data.try_cast < std::vector<torch::Tensor>>()) {
    std::vector<torch::Tensor> tensors = opt.value();
    return tensors.at(0).is_cpu();
    }

  return true;
}

inline c10::cuda::CUDAStream get_current_stream() {
  return c10::cuda::getCurrentCUDAStream();
}

// GPU事件初始化（线程安全版）
inline const at::cuda::CUDAEvent& start_event() {
  static at::cuda::CUDAEvent ev;
  static std::once_flag flag;
  std::call_once(flag, [&] { ev.record(at::cuda::getDefaultCUDAStream()); });
  return ev;
}

// 获取当前CUDA流的时间（毫秒），对齐CPU时间
float cuda_time() {
  // 记录GPU结束事件
  at::cuda::CUDAEvent stop_event;
  stop_event.record(get_current_stream());
  stop_event.synchronize();

  // 计算GPU时间
  float gpu_ms = start_event().elapsed_time(stop_event);

  // 初始化时间偏移量（只执行一次）
  static float time_offset = [&]() {
    start_event().synchronize();
    at::cuda::CUDAEvent sync_event;
    sync_event.record(get_current_stream());
    sync_event.synchronize();

    auto cpu_elapsed = omniback::helper::timestamp();
    return cpu_elapsed - start_event().elapsed_time(sync_event);
  }();

  // 返回对齐后的时间
  return gpu_ms + time_offset;
}

/**
 * @brief switch_device async only for pinned_memory2gpu
 *
 * @note
https://docs.nvidia.com/cuda/cuda-runtime-api/api-sync-behavior.html#api-sync-behavior
https://discuss.pytorch.org/t/should-we-set-non-blocking-to-true/38234/16
 * @param input
 * @return torch::Tensor
 */
torch::Tensor switch_device(torch::Tensor input) {
  if (input.device() == torch::kCUDA)
    return input.cpu();
  torch::TensorOptions options;
  assert(input.device() == torch::kCPU);

  if (!input.is_pinned())
    return input.cuda();
  return input.to(
      torch::TensorOptions().device(torch::kCUDA, -1),
      false,
      false,
      torch::MemoryFormat::Contiguous); // 这里为异步操作, pytorch 自身cache
                                        // pinned memory， 不怕析构
}

torch::Tensor async2cpu(torch::Tensor input) {
  //  auto options=torch::TensorOptions()
  //   .device(torch::kCPU)
  //   .dtype(featRegion.dtype()) // torch::kByte
  //   .layout(torch::kStrided)
  //   .requires_grad(false).pinned_memory(false);
  torch::TensorOptions options;
  if (input.device() == torch::kCPU) {
    return input;
  } else
    options = torch::TensorOptions().device(torch::kCPU).pinned_memory(true);

  return input.to(
      options,
      true,
      false,
      torch::MemoryFormat::Contiguous); // 这里为异步操作
}

bool is_channel(torch::Tensor in, unsigned right_index, unsigned wrong_index) {
  if (right_index > 0 && right_index >= in.sizes().size()) {
    return false;
  } else if (wrong_index > 0 && wrong_index >= in.sizes().size()) {
    return false;
  }
  if ((in.size(right_index) == 1 || in.size(right_index) == 3) &&
      (in.size(wrong_index) != 1 && in.size(wrong_index) != 3)) {
    return true;
  }
  return false;
}
torch::Tensor tensor2nchw(torch::Tensor in) {
  torch::Tensor target;
  bool shape_ok = true;
  if (in.sizes().size() == 3) {
    if (is_channel(in, 0, 2)) { // chw
      target = in.unsqueeze(0);
    } else if (is_channel(in, 2, 0)) { // hwc
      target = in.permute({2, 0, 1}).unsqueeze(0);
    } else {
      shape_ok = false;
      // error
    }

  } else if (in.sizes().size() == 4) {
    if (is_channel(in, 1, 3)) { // nchw
      target = in;
    } else if (is_channel(in, 3, 1)) { // nhwc
      target = in.permute({0, 3, 1, 2});
    } else {
      shape_ok = false;
    }
  }
  if (!shape_ok) {
    SPDLOG_ERROR("shape inference failed");
    throw std::runtime_error("shape inference failed. Maybe too small");
  }
  return target;
}

torch::Tensor img_1chw_guard(torch::Tensor in) {
  const auto& in_size = in.sizes();

  if (in_size.size() == 3 &&
      (in_size[2] == 1 || in_size[2] == 3 || in_size[2] == 4) &&
      (in_size[0] * in_size[1] != 0)) { // hwc
    return in.permute({2, 0, 1}).unsqueeze(0);
  } else if (
      in_size.size() == 4 &&
      (in_size[1] == 1 || in_size[1] == 3 || in_size[1] == 4) &&
      in_size[0] == 1 && (in_size[2] * in_size[3] != 0)) { // 1chw
    return in;
  } else {
    std::stringstream out;
    out << "Only support 1chw or hwc, with c == 1, 3, 4; But the shape of "
           "input Tensor is ";
    out << in_size;
    SPDLOG_ERROR(out.str());
    throw std::out_of_range(out.str());
  }
}

torch::Tensor img_nchw_guard(torch::Tensor in) {
  const auto& in_size = in.sizes();

  if (in_size.size() == 3 &&
      (in_size[2] == 1 || in_size[2] == 3 || in_size[2] == 4) &&
      (in_size[0] * in_size[1] != 0)) { // hwc
    return in.permute({2, 0, 1}).unsqueeze(0);
  } else if (
      in_size.size() == 4 &&
      (in_size[1] == 1 || in_size[1] == 3 || in_size[1] == 4) &&
      (in_size[2] * in_size[3] != 0)) { // nchw
    return in;
  } else {
    std::stringstream out;
    out << "Only support nchw or hwc, with c == 1, 3, 4; But the shape of "
           "input Tensor is ";
    out << in_size;
    SPDLOG_ERROR(out.str());
    throw std::out_of_range(out.str());
  }
}

torch::Tensor img_1hwc_guard(torch::Tensor in) {
  const auto& in_size = in.sizes();

  if (in_size.size() == 3 &&
      (in_size[2] == 1 || in_size[2] == 3 || in_size[2] == 4) &&
      (in_size[0] * in_size[1] != 0)) { // hwc
    return in.unsqueeze(0);
  } else if (
      in_size.size() == 4 &&
      (in_size[1] == 1 || in_size[1] == 3 || in_size[1] == 4) &&
      in_size[0] == 1 && (in_size[2] * in_size[3] != 0)) { // 1chw
    return in.permute({0, 2, 3, 1});
  } else {
    std::stringstream out;
    out << "Only support 1chw or hwc, with c == 1, 3, 4; But the shape of "
           "input Tensor is ";
    out << in_size;
    SPDLOG_ERROR(out.str());
    throw std::out_of_range(out.str());
  }
}

bool is_1chw(torch::Tensor in) {
  const auto& in_size = in.sizes();

  if (in_size.size() == 4 &&
      (in_size[1] == 1 || in_size[1] == 3 || in_size[1] == 4) &&
      in_size[0] == 1 && (in_size[2] * in_size[3] != 0)) { // 1chw
    return true;
  }
  return false;
}

bool is_contiguous_wrt_nchw(torch::Tensor in) {
  if (is_nchw(in)) {
    if (in.is_contiguous())
      return true;
    else
      return false;
  } else if (is_hwc(in)) {
    auto tmp = in.permute({2, 0, 1});
    if (tmp.is_contiguous())
      return true;
    else
      return false;
  }
  return false;
}

bool is_contiguous_wrt_hwc(torch::Tensor in) {
  if (is_hwc(in)) {
    if (in.is_contiguous())
      return true;
    else
      return false;
  } else if (is_1chw(in)) {
    auto tmp = in.permute({0, 2, 3, 1});
    if (tmp.is_contiguous())
      return true;
    else
      return false;
  }
  return false;
}

bool is_nchw(torch::Tensor in) {
  const auto& in_size = in.sizes();

  if (in_size.size() == 4 &&
      (in_size[1] == 1 || in_size[1] == 3 || in_size[1] == 4) &&
      (in_size[2] * in_size[3] != 0)) { // nchw
    return true;
  }
  return false;
}

bool is_hwc(torch::Tensor in) {
  const auto& in_size = in.sizes();

  if (in_size.size() == 3 &&
      (in_size[2] == 1 || in_size[2] == 3 || in_size[2] == 4) &&
      (in_size[0] * in_size[1] != 0)) { // hwc
    return true;
  }
  return false;
}

MemoryFormat guard_valid_memory_format(const torch::Tensor& data) {
  const auto& in_size = data.sizes();

  if (in_size.size() == 3 &&
      (in_size[2] == 1 || in_size[2] == 3 || in_size[2] == 4) &&
      (in_size[0] * in_size[1] != 0)) { // hwc
    return MemoryFormat::HWC;
  } else if (
      in_size.size() == 4 &&
      (in_size[1] == 1 || in_size[1] == 3 || in_size[1] == 4) &&
      (in_size[2] * in_size[3] != 0)) { // nchw
    return MemoryFormat::NCHW;
  }
  throw std::invalid_argument("Only hwc and nchw are supported.");
}

torch::Tensor img_hwc_guard(torch::Tensor in) {
  const auto& in_size = in.sizes();

  if (in_size.size() == 3 &&
      (in_size[2] == 1 || in_size[2] == 3 || in_size[2] == 4) &&
      (in_size[0] * in_size[1] != 0)) { // hwc
    return in;
  } else if (
      in_size.size() == 4 &&
      (in_size[1] == 1 || in_size[1] == 3 || in_size[1] == 4) &&
      in_size[0] == 1 && (in_size[2] * in_size[3] != 0)) { // 1chw
    return in.permute({2, 3, 1, 0}).squeeze(3);
  } else {
    std::stringstream out;
    out << "Only support 1chw or hwc, with c == 1, 3, 4; But the shape of "
           "input Tensor is ";
    out << in_size;
    SPDLOG_ERROR(out.str());
    throw std::out_of_range(out.str());
  }
}

torch::Tensor tensor_permute(
    torch::Tensor input,
    const std::vector<int>& min_shape,
    const std::vector<int>& max_shape,
    bool& need_permute) {
  if (max_shape.size() != min_shape.size()) {
    throw std::runtime_error("max_shape.size() != min_shape.size()");
  }
  if (input.sizes().size() == max_shape.size() - 1) {
    input = input.unsqueeze(0);
  } else if (
      input.sizes().size() == max_shape.size() + 1 && input.size(0) == 1) {
    input = input.squeeze(0);
  } else if (input.sizes().size() == max_shape.size()) {
    if (input.sizes()[0] > max_shape[0]) {
      std::stringstream ss;
      ss << "data's batchsize(" << input.sizes()[0]
         << ") should be smaller than max batch size(";
      for (const auto item : max_shape)
        ss << item << " ";
      ss << ")";
      throw std::runtime_error(ss.str());
    }
  } else {
    throw std::runtime_error(
        "input data's dim not match model's. input = " +
        std::to_string(input.sizes().size()) +
        ", model = " + std::to_string(max_shape.size()));
  }
  need_permute = false;
  std::vector<int64_t> permute_vector{0};
  for (std::size_t i = 1; i < max_shape.size(); ++i) {
    if (max_shape[i] < input.sizes()[i] || min_shape[i] > input.sizes()[i] ||
        need_permute) {
      need_permute = true;
      for (std::size_t j = 1; j < max_shape.size(); ++j) {
        if (max_shape[i] >= input.sizes()[j] &&
            min_shape[i] <= input.sizes()[j] &&
            std::find(permute_vector.begin(), permute_vector.end(), j) ==
                permute_vector.end()) {
          permute_vector.push_back(j);
          break;
        }
      }
    } else {
      permute_vector.push_back(i);
    }
  }
  if (need_permute) {
    if (permute_vector.size() != max_shape.size()) {
      std::stringstream ss;
      ss << "network's max = ";
      for (const auto& item : max_shape)
        ss << std::to_string(item) << ",";
      ss << " input data = ";
      for (const auto& item : input.sizes())
        ss << std::to_string(item) << ",";
      ss << " please check it.";
      throw std::runtime_error(
          "input shape and network shape not match. " + ss.str());
    }
    // SPDLOG_DEBUG("implicit batch mode deprecated.");
    input = input.permute(permute_vector);
  }
  return input;
}

std::vector<torch::Tensor> get_tensors(
    omniback::dict input_dict,
    const std::string& key) {
  auto iter = input_dict->find(key);
  OMNI_ASSERT(iter != input_dict->end(), "key not found: " + key);
  std::vector<torch::Tensor> image_embeds;
  omniback::any& data = iter->second;
  if (auto opt = data.try_cast<torch::Tensor>()) {
    torch::Tensor input_tensor = opt.value();

    image_embeds.push_back(input_tensor);
  } else if (auto opt = data.try_cast<std::vector<torch::Tensor>>()) {
    image_embeds = opt.value();
    }
  else {
    TVM_FFI_THROW(TypeError)<< "get_tensors: input is not a tensor or a list of tensors.";
  }
  return image_embeds;
}

torch::Tensor try_quick_cat(std::vector<torch::Tensor> resized_inputs) {
  IPIPE_ASSERT(resized_inputs.size() >= 2);
  bool share_same_storage = true;
  bool is_continuous = true;
  auto first_data_ptr = resized_inputs[0].storage().data_ptr().get();
  auto last_offset =
      resized_inputs[0].storage_offset() + resized_inputs[0].numel();

  // Calculate total size
  int64_t total_size = resized_inputs[0].numel();

  for (size_t i = 1; i < resized_inputs.size(); ++i) {
    const auto& tensor = resized_inputs[i];
    assert(tensor.is_contiguous());
    total_size += tensor.numel();
    if (tensor.storage().data_ptr().get() != first_data_ptr) {
      share_same_storage = false;
      break;
    }
    if (tensor.storage_offset() != last_offset) {
      is_continuous = false;
      break;
    }
    last_offset += tensor.numel();
  }

  torch::Tensor true_input;
  if (share_same_storage && is_continuous) {
    // All tensors share the same storage and they are continuous.
    // You can reuse the storage.

    auto sizes = resized_inputs[0].sizes().vec();
    sizes[0] = resized_inputs.size();
    true_input = torch::empty({0}, resized_inputs[0].options())
                     .set_(
                         resized_inputs[0].storage(),
                         resized_inputs[0].storage_offset(),
                         sizes,
                         resized_inputs[0].strides());
    IPIPE_ASSERT(true_input.is_contiguous());
    SPDLOG_DEBUG("use quick cat");
    assert(true_input.storage().data_ptr().get() == first_data_ptr);
  } else {
    // Tensors do not share the same storage or they are not continuous.
    // You need to concatenate them.
    true_input = torch::cat(resized_inputs, 0);
  }

  return true_input;
}

std::string get_sm() {
  cudaDeviceProp* prop = at::cuda::getCurrentDeviceProperties();
  return std::to_string(prop->major) + "." + std::to_string(prop->minor);
}

// Function to ensure the tensor is of the desired type
void fix_tensor_type(torch::Tensor& input, NetIOInfo::DataType in_type) {
  auto desired_type = netinfo2torch_type(in_type);
  if (input.dtype() != desired_type) {
    input = input.to(desired_type);
  }
}

// Function to ensure the tensor is on the desired device
void fix_tensor_device(torch::Tensor& input, NetIOInfo::Device in_device) {
  auto desired_device = netinfo2torch_device(in_device);
  if (input.device() != desired_device) {
    input = input.to(desired_device);
  }
}

void fix_tensor_shape(
    torch::Tensor& data,
    const NetIOInfo::Dims64 min,
    const NetIOInfo::Dims64& max) {
  const auto& sizes = data.sizes();

  if (sizes.size() == 3 && 4 == min.nbDims && max.d[1] == min.d[1] &&
      min.d[1] <= 4) {
    // hwc2nchw
    if ((sizes[0] >= min.d[2] && sizes[0] <= max.d[2]) &&
        sizes[1] >= min.d[3] && sizes[1] <= max.d[3] && sizes[2] == min.d[1]) {
      if ((sizes[0] == min.d[1]) && sizes[1] >= min.d[2] &&
          sizes[1] <= max.d[2] && sizes[2] >= min.d[3] &&
          sizes[2] <= max.d[3]) {
        // chw
        throw std::invalid_argument(
            "fix_tensor_shape: Cannot handle ambiguity. "
            "The input tensor can be interpreted as either HWC or "
            "NCHW.");
      }
      data = data.permute({2, 0, 1}).unsqueeze(0);
      return;
    }
  }

  // bool in_error = false;
  std::string err_msg;
  if (sizes.size() == min.nbDims) {
    // todo: after cat
    // for (size_t i = 0; i < sizes.size(); ++i) {
    //   if (sizes[i] < min.d[i] || sizes[i] > max.d[i]) {
    //     // in_error = true;
    //     err_msg = std::to_string(sizes[i]) + " is not in range [" +
    //         std::to_string(min.d[i]) + ", " + std::to_string(max.d[i]) +
    //         "]. index=" + std::to_string(i);
    //     break;
    //   }
    // }
  } else if (sizes.size() + 1 == min.nbDims) {
    if ((sizes[0] >= min.d[1] && sizes[0] <= max.d[1]) &&
        sizes[1] >= min.d[2] && sizes[1] <= max.d[2] && sizes[2] >= min.d[3] &&
        sizes[2] <= max.d[3]) {
      data = data.unsqueeze(0);
      return;
    }
  }
  if (!err_msg.empty())
    throw std::invalid_argument(
        "fix_tensor_shape: invalid tensor shape : " +
        omniback::str::vec2str(data.sizes().vec()) + " err_msg: " + err_msg);
}

// Function to convert NetIOInfo::Device to torch::Device
torch::Device netinfo2torch_device(NetIOInfo::Device device) {
  switch (device) {
    case NetIOInfo::Device::CPU:
      return torch::Device(torch::kCPU);
    case NetIOInfo::Device::GPU:
      return torch::Device(torch::kCUDA);
    default:
      throw std::invalid_argument("Unknown device type");
  }
}

c10::ScalarType netinfo2torch_type(NetIOInfo::DataType dtype) {
  switch (dtype) {
    case NetIOInfo::DataType::INT8:
      return torch::kInt8;
    case NetIOInfo::DataType::UINT8:
      return torch::kUInt8;
    case NetIOInfo::DataType::INT32:
      return torch::kInt32;
    case NetIOInfo::DataType::INT64:
      return torch::kInt64;
    case NetIOInfo::DataType::BOOL:
      return torch::kBool;
    case NetIOInfo::DataType::FP16:
      return torch::kFloat16;
    case NetIOInfo::DataType::FP32:
      return torch::kFloat32;
    case NetIOInfo::DataType::BF16:
      return torch::kBFloat16;
    case NetIOInfo::DataType::BF32:
    case NetIOInfo::DataType::INT4:
    case NetIOInfo::DataType::FP4:
    case NetIOInfo::DataType::FP8:
    case NetIOInfo::DataType::RESERVED_INT:
    case NetIOInfo::DataType::RESERVED_FP:
    case NetIOInfo::DataType::RESERVED_BF:
    case NetIOInfo::DataType::UNKNOWN:
    default:
      throw std::runtime_error("Unsupported or unknown data type");
  }
}
std::string print_torch_scale_type(c10::ScalarType tp) {
  switch (tp) {
    case torch::kInt8:
      return "Int8";
    case torch::kUInt8:
      return "UInt8";
    case torch::kInt16:
      return "Int16";
    case torch::kInt32:
      return "Int32";
    case torch::kInt64:
      return "Int64";
    case torch::kFloat16:
      return "Float16";
    case torch::kFloat32:
      return "Float32";
    case torch::kFloat64:
      return "Float64";
    case torch::kBool:
      return "Bool";
    case torch::kBFloat16:
      return "BFloat16";
    case torch::kQInt8:
      return "QInt8";
    case torch::kQUInt8:
      return "QUInt8";
    case torch::kQInt32:
      return "QInt32";
    case torch::kComplexFloat:
      return "ComplexFloat";
    case torch::kComplexDouble:
      return "ComplexDouble";
    default:
      return "Unknown";
  }
}
void fix_tensors(
    std::vector<torch::Tensor>& tensors,
    const std::shared_ptr<NetIOInfos>& infos) {
  const auto num_inputs = tensors.size();
  OMNI_ASSERT(
      infos->first.size() >= num_inputs,
      "number of inputs from model does not match that from the data");

  for (size_t i = 0; i < num_inputs; ++i) {
    fix_tensor_shape(
        tensors[i], infos->first.at(i).min, infos->first.at(i).max);
    fix_tensor_type(tensors[i], infos->first.at(i).type);
    fix_tensor_device(tensors[i], infos->first.at(i).device);
  }
}
std::string print_tensor(
    const std::vector<torch::Tensor>& data,
    const std::string& tag) {
  std::ostringstream oss;
  for (size_t i = 0; i < data.size(); ++i) {
    if (!tag.empty()) {
      oss << "tag = " << tag << ". ";
    }
    oss << "Tensor " << i << " shape = " << data[i].sizes() << "\n";
  }

  for (const auto& item : data) {
    if (item.numel() > 60) {
      auto new_view = item.view(-1); // 将张量展平
      auto head = new_view.slice(0, 0, 5); // 取前5个元素
      auto tail = new_view.slice(0, -5, new_view.size(0)); // 取后5个元素
      oss << "Tensor is large. Shape: " << item.sizes()
          << ". Showing head and tail:\n";
      oss << head << "\n...\n" << tail << "\n";
    } else {
      oss << item << "\n";
    }
  }
  return oss.str();
}

void check_input_tensor(const torch::Tensor& tensor, const NetIOInfo& infos) {
  if (tensor.sizes().size() != infos.min.nbDims) {
    OMNI_ASSERT(
        false,
        std::to_string(tensor.sizes().size()) + " vs " +
            std::to_string(infos.min.nbDims) + "\n" + print_tensor({tensor}));
  }

  for (size_t i = 0; i < infos.min.nbDims; ++i) {
    if (tensor.sizes()[i] < infos.min.d[i]) {
      std::ostringstream oss;
      oss << "Input tensor shape does not match the min shape required by the model. "
          << "tensor.sizes() = " << tensor.sizes() << ", infos.min.d[" << i
          << "] = " << infos.min.d[i] << "\n";
      OMNI_ASSERT(false, oss.str());
    }

    OMNI_ASSERT(
        tensor.sizes()[i] <= infos.max.d[i],
        "tensor.sizes()[i] = " + std::to_string(tensor.sizes()[i]) +
            " infos.max.d[i] = " + std::to_string(infos.max.d[i]));
    OMNI_ASSERT(
        tensor.scalar_type() == netinfo2torch_type(infos.type),
        "Input tensor data type does not match the data type "
        "required by the model. " +
            print_torch_scale_type(tensor.scalar_type()) + " vs " +
            print_torch_scale_type(netinfo2torch_type(infos.type)));
    OMNI_ASSERT(
        tensor.is_cuda() & (infos.device == NetIOInfo::Device::GPU),
        std::to_string(tensor.is_cuda()) + " vs " +
            std::to_string(infos.device == NetIOInfo::Device::GPU)); // todo
    OMNI_ASSERT(tensor.is_contiguous());
  }
}

void check_batched_inputs(
    const std::vector<torch::Tensor>& tensors,
    const std::vector<NetIOInfo>& infos) {
  // SPDLOG_DEBUG(print_tensor(tensors, "check_batched_inputs"));
  const auto num_inputs = tensors.size();
  OMNI_ASSERT(
      infos.size() == num_inputs,
      "number of inputs from model does not match that from the data");

  for (size_t i = 0; i < num_inputs; ++i) {
    check_input_tensor(tensors[i], infos[i]);
  }
}

bool match(NetIOInfo::Dims64& dst, const torch::Tensor& src) {
  if (dst.nbDims != src.sizes().size())
    return false;
  bool shape_is_match = true;
  for (size_t i = 0; i < dst.nbDims; ++i) {
    if (dst.d[i] != src.sizes()[i]) {
      dst.d[i] = src.sizes()[i];
      shape_is_match = false; // no break!
    }
  }
  return shape_is_match;
}

// int StreamOrderedManagedTensorAllocator(
//     void* stream,
//     DLTensor* prototype,
//     DLManagedTensorVersioned** out,
//     void* error_ctx,
//     void (*SetError)(void* error_ctx, const char* kind, const char* message)) {

//   at::cuda::CUDAStream cuda_stream = at::cuda::getStreamFromExternal(
//       static_cast<cudaStream_t>(stream), c10::cuda::current_device());

//   static DLPackManagedTensorAllocator& alloc = torch_allocator();
//   return alloc(prototype, out, error_ctx, SetError);
// }


} // namespace torchpipe