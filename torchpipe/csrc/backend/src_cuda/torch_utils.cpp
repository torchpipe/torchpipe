// Copyright 2021-2024 NetEase.
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

#include "torch_utils.hpp"

#include <mutex>
#include <atomic>
#include <chrono>
#include <thread>
#ifdef WITH_CUDA
#include "c10/cuda/CUDAStream.h"
#endif
#include "time_utils.hpp"
#include "base_logging.hpp"
#include <torch/serialize.h>
#include <torch/extension.h>
#include <torch/torch.h>
#include <fstream>

namespace ipipe {

void save(std::string save_name, torch::Tensor input) {
  std::vector<char> data_for_save = torch::pickle_save(input);
  // save_name
  std::ofstream fout(save_name, std::ios::out | std::ios::binary);
  fout.write(data_for_save.data(), data_for_save.size());
  fout.close();
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

#ifdef WITH_CUDA
bool torch_not_use_default_stream(bool high_prio) {
  if (c10::cuda::getCurrentCUDAStream() == c10::cuda::getDefaultCUDAStream()) {
    c10::cuda::setCurrentCUDAStream(
        c10::cuda::getStreamFromPool(high_prio,
                                     -1));  // Schedule保证了init和forward在同一个线程
    return true;
  }
  return false;
}

bool torch_not_use_default_stream(int device_id, bool high_prio) {
  if (c10::cuda::current_device() != device_id && device_id >= 0) {
    c10::cuda::set_device(device_id);
  }
  if (c10::cuda::getCurrentCUDAStream(device_id) == c10::cuda::getDefaultCUDAStream(device_id)) {
    c10::cuda::setCurrentCUDAStream(c10::cuda::getStreamFromPool(
        high_prio, device_id));  // Schedule保证了init和forward在同一个线程
    return true;
  }
  return false;
}

bool torch_is_using_default_stream() {
  if (c10::cuda::getCurrentCUDAStream(-1) == c10::cuda::getDefaultCUDAStream(-1)) {
    return true;
  }
  return false;
}

// https://discuss.pytorch.org/t/asynchronous-copy-in-c-when-input-has-been-destructed/186515
torch::Tensor to_current_device(torch::Tensor input) {
  if (input.device() == torch::kCPU) return input.cuda();
  if (input.device().index() == c10::cuda::current_device()) return input;
  torch::TensorOptions options;
  // input.is_pinned()
  return input.to(torch::TensorOptions().device(torch::kCUDA, -1), false, false,
                  input.suggest_memory_format());  // 这里为异步操作, pytorch 自身cache pinned
                                                   // memory， 不怕析构
}
#endif

bool is_cpu_tensor(torch::Tensor input) {
#ifndef TORCH_VERSION_MAJOR
  IPIPE_CHECK(0, "TORCH_VERSION_MAJOR not defined");
#endif
#if TORCH_VERSION_MAJOR == 1 && TORCH_VERSION_MINOR < 10
  return input.device().is_cpu();
#else
  return input.is_cpu();
#endif
}
// Check if the given data variable is of CPU type.
bool is_any_cpu(any data) {
  if (data.type() == typeid(torch::Tensor)) {
    torch::Tensor tensor = any_cast<torch::Tensor>(data);
    return is_cpu_tensor(tensor);
  }

  if (data.type() == typeid(std::vector<torch::Tensor>)) {
    std::vector<torch::Tensor> tensors = any_cast<std::vector<torch::Tensor>>(data);
    return tensors.at(0).is_cpu();
  }

  return true;
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
  if (input.device() == torch::kCUDA) return input.cpu();
  torch::TensorOptions options;
  assert(input.device() == torch::kCPU);

  if (!input.is_pinned()) return input.cuda();
  return input.to(torch::TensorOptions().device(torch::kCUDA, -1), false, false,
                  torch::MemoryFormat::Contiguous);  // 这里为异步操作, pytorch 自身cache pinned
                                                     // memory， 不怕析构
}

//  测试 cudaMemcpyAsync 是同步还是异步（非pinnedmemory）
// 结论 此时 cudaMemcpyAsync 和 cudaMemcpy 差不多
class TestRun {
 public:
  TestRun() {
    // test_cudaMemcpyAsync();
  };

  void test_run(int src[]) {
    while (data_ >= 0) {
      std::lock_guard<std::mutex> lock(lock_);
      src[99] = ++data_;
    }
  }

  // void test_cudaMemcpyAsync() {
  //   int* src = new int[1000000];

  //   int final_data[100]{0};

  //   int* device_data = NULL;
  //   size_t size = 1000000 * sizeof(int);
  //   cudaMalloc((void**)&device_data, size);
  //   auto stream = c10::cuda::getCurrentCUDAStream();
  //   stream.synchronize(); // todo 此处 d

  //   std::thread th(&TestRun::test_run, this, src);
  //   std::this_thread::sleep_for(std::chrono::milliseconds(10));

  //   {
  //     std::lock_guard<std::mutex> lock(lock_);
  //     std::cout << "before cudaMemcpyAsync data_= " << data_ << " src[99] "
  //               << src[99] << std::endl;
  //     {
  //       time_guard z("cudaMemcpyAsync");
  //       // cudaMemcpyAsync(device_data, src, 1000000*sizeof(int),
  //       // cudaMemcpyHostToDevice, stream);
  //       cudaMemcpy(
  //           device_data, src, 1000000 * sizeof(int), cudaMemcpyHostToDevice);
  //     }
  //     if (true) {
  //       delete src;
  //       {
  //         time_guard z("synchronize");
  //         stream.synchronize(); // todo 此处 d
  //       }

  //       std::this_thread::sleep_for(std::chrono::milliseconds(1000));
  //       std::cout << "finish" << std::endl;
  //       ;
  //       std::abort();
  //     }
  //     src[99] += 1;
  //   }

  //   std::cout << "after cudaMemcpyAsync data_= " << data_ << " src[99] "
  //             << src[99] << std::endl;

  //   stream.synchronize(); // todo 此处 d
  //   cudaMemcpyAsync(
  //       final_data,
  //       device_data,
  //       100 * sizeof(int),
  //       cudaMemcpyDeviceToHost,
  //       stream);
  //   stream.synchronize(); // todo 此处 d
  //   data_ = -100;
  //   std::cout << " stop | data_= " << data_ << " src[99] = " << src[99]
  //             << " final_data[99]=" << final_data[99] << std::endl;

  //   th.join();
  //   std::abort();
  // }
  std::atomic<int> data_{1};
  std::mutex lock_;
};
// TestRun tmp;

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

  return input.to(options, true, false, torch::MemoryFormat::Contiguous);  // 这里为异步操作
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
    if (is_channel(in, 0, 2)) {  // chw
      target = in.unsqueeze(0);
    } else if (is_channel(in, 2, 0)) {  // hwc
      target = in.permute({2, 0, 1}).unsqueeze(0);
    } else {
      shape_ok = false;
      // error
    }

  } else if (in.sizes().size() == 4) {
    if (is_channel(in, 1, 3)) {  // nchw
      target = in;
    } else if (is_channel(in, 3, 1)) {  // nhwc
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

  if (in_size.size() == 3 && (in_size[2] == 1 || in_size[2] == 3 || in_size[2] == 4) &&
      (in_size[0] * in_size[1] != 0)) {  // hwc
    return in.permute({2, 0, 1}).unsqueeze(0);
  } else if (in_size.size() == 4 && (in_size[1] == 1 || in_size[1] == 3 || in_size[1] == 4) &&
             in_size[0] == 1 && (in_size[2] * in_size[3] != 0)) {  // 1chw
    return in;
  } else {
    std::stringstream out;
    out << "Only support 1chw or hwc, with c == 1, 3, 4; But the shape of input Tensor is ";
    out << in_size;
    SPDLOG_ERROR(out.str());
    throw std::out_of_range(out.str());
  }
}

torch::Tensor img_nchw_guard(torch::Tensor in) {
  const auto& in_size = in.sizes();

  if (in_size.size() == 3 && (in_size[2] == 1 || in_size[2] == 3 || in_size[2] == 4) &&
      (in_size[0] * in_size[1] != 0)) {  // hwc
    return in.permute({2, 0, 1}).unsqueeze(0);
  } else if (in_size.size() == 4 && (in_size[1] == 1 || in_size[1] == 3 || in_size[1] == 4) &&
             (in_size[2] * in_size[3] != 0)) {  // nchw
    return in;
  } else {
    std::stringstream out;
    out << "Only support nchw or hwc, with c == 1, 3, 4; But the shape of input Tensor is ";
    out << in_size;
    SPDLOG_ERROR(out.str());
    throw std::out_of_range(out.str());
  }
}

torch::Tensor img_1hwc_guard(torch::Tensor in) {
  const auto& in_size = in.sizes();

  if (in_size.size() == 3 && (in_size[2] == 1 || in_size[2] == 3 || in_size[2] == 4) &&
      (in_size[0] * in_size[1] != 0)) {  // hwc
    return in.unsqueeze(0);
  } else if (in_size.size() == 4 && (in_size[1] == 1 || in_size[1] == 3 || in_size[1] == 4) &&
             in_size[0] == 1 && (in_size[2] * in_size[3] != 0)) {  // 1chw
    return in.permute({0, 2, 3, 1});
  } else {
    std::stringstream out;
    out << "Only support 1chw or hwc, with c == 1, 3, 4; But the shape of input Tensor is ";
    out << in_size;
    SPDLOG_ERROR(out.str());
    throw std::out_of_range(out.str());
  }
}

bool is_1chw(torch::Tensor in) {
  const auto& in_size = in.sizes();

  if (in_size.size() == 4 && (in_size[1] == 1 || in_size[1] == 3 || in_size[1] == 4) &&
      in_size[0] == 1 && (in_size[2] * in_size[3] != 0)) {  // 1chw
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

  if (in_size.size() == 4 && (in_size[1] == 1 || in_size[1] == 3 || in_size[1] == 4) &&
      (in_size[2] * in_size[3] != 0)) {  // nchw
    return true;
  }
  return false;
}

bool is_hwc(torch::Tensor in) {
  const auto& in_size = in.sizes();

  if (in_size.size() == 3 && (in_size[2] == 1 || in_size[2] == 3 || in_size[2] == 4) &&
      (in_size[0] * in_size[1] != 0)) {  // hwc
    return true;
  }
  return false;
}

torch::Tensor img_hwc_guard(torch::Tensor in) {
  const auto& in_size = in.sizes();

  if (in_size.size() == 3 && (in_size[2] == 1 || in_size[2] == 3 || in_size[2] == 4) &&
      (in_size[0] * in_size[1] != 0)) {  // hwc
    return in;
  } else if (in_size.size() == 4 && (in_size[1] == 1 || in_size[1] == 3 || in_size[1] == 4) &&
             in_size[0] == 1 && (in_size[2] * in_size[3] != 0)) {  // 1chw
    return in.permute({2, 3, 1, 0}).squeeze(3);
  } else {
    std::stringstream out;
    out << "Only support 1chw or hwc, with c == 1, 3, 4; But the shape of input Tensor is ";
    out << in_size;
    SPDLOG_ERROR(out.str());
    throw std::out_of_range(out.str());
  }
}

torch::Tensor tensor_permute(torch::Tensor input, const std::vector<int>& min_shape,
                             const std::vector<int>& max_shape, bool& need_permute) {
  if (max_shape.size() != min_shape.size()) {
    throw std::runtime_error("max_shape.size() != min_shape.size()");
  }
  if (input.sizes().size() == max_shape.size() - 1) {
    input = input.unsqueeze(0);
  } else if (input.sizes().size() == max_shape.size()) {
    if (input.sizes()[0] > max_shape[0]) {
      std::stringstream ss;
      ss << "data's batchsize(" << input.sizes()[0] << ") should be smaller than max batch size("
         << max_shape[0] << ")";
      throw std::runtime_error(ss.str());
    }
  } else {
    throw std::runtime_error("input data's dim not match model's. input = " +
                             std::to_string(input.sizes().size()));
  }
  need_permute = false;
  std::vector<int64_t> permute_vector{0};
  for (std::size_t i = 1; i < max_shape.size(); ++i) {
    if (max_shape[i] < input.sizes()[i] || min_shape[i] > input.sizes()[i] || need_permute) {
      need_permute = true;
      for (std::size_t j = 1; j < max_shape.size(); ++j) {
        if (max_shape[i] >= input.sizes()[j] && min_shape[i] <= input.sizes()[j] &&
            std::find(permute_vector.begin(), permute_vector.end(), j) == permute_vector.end()) {
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
      for (const auto& item : max_shape) ss << std::to_string(item) << "x";
      ss << "input data = ";
      for (const auto& item : input.sizes()) ss << std::to_string(item) << "x";
      ss << "please check it.";
      throw std::runtime_error("input shape and network shape not match. " + ss.str());
    }
    // SPDLOG_DEBUG("implicit batch mode deprecated.");
    input = input.permute(permute_vector);
  }
  return input;
}

torch::Tensor try_quick_cat(std::vector<torch::Tensor> resized_inputs) {
  IPIPE_ASSERT(resized_inputs.size() >= 2);
  bool share_same_storage = true;
  bool is_continuous = true;
  auto first_data_ptr = resized_inputs[0].storage().data_ptr().get();
  auto last_offset = resized_inputs[0].storage_offset() + resized_inputs[0].numel();

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
                     .set_(resized_inputs[0].storage(), resized_inputs[0].storage_offset(), sizes,
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

}  // namespace ipipe