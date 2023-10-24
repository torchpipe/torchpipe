// Copyright 2021-2023 NetEase.
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
#include "c10/cuda/CUDAStream.h"
#include "time_utils.hpp"
#include "base_logging.hpp"
#include <torch/serialize.h>
#include <torch/extension.h>
#include <ATen/ATen.h>

namespace ipipe {

void save(std::string save_name, at::Tensor input) {
  std::vector<char> data_for_save = torch::pickle_save(input);
  // save_name
  std::ofstream fout(save_name, std::ios::out | std::ios::binary);
  fout.write(data_for_save.data(), data_for_save.size());
  fout.close();
}

at::Tensor load_tensor(std::string save_name) {
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

bool is_cpu_tensor(at::Tensor input) {
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
  if (data.type() == typeid(at::Tensor)) {
    at::Tensor tensor = any_cast<at::Tensor>(data);
    return is_cpu_tensor(tensor);
  }

  if (data.type() == typeid(std::vector<at::Tensor>)) {
    std::vector<at::Tensor> tensors = any_cast<std::vector<at::Tensor>>(data);
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
 * @return at::Tensor
 */
at::Tensor switch_device(at::Tensor input) {
  if (input.device() == at::kCUDA) return input.cpu();
  at::TensorOptions options;
  assert(input.device() == at::kCPU);

  if (!input.is_pinned()) return input.cuda();
  return input.to(at::TensorOptions().device(at::kCUDA, -1), false, false,
                  at::MemoryFormat::Contiguous);  // 这里为异步操作, pytorch 自身cache pinned
                                                  // memory， 不怕析构
}

// https://discuss.pytorch.org/t/asynchronous-copy-in-c-when-input-has-been-destructed/186515
at::Tensor to_current_device(at::Tensor input) {
  if (input.device() == at::kCPU) return input.cuda();
  if (input.device().index() == at::cuda::current_device()) return input;
  at::TensorOptions options;
  // input.is_pinned()
  return input.to(at::TensorOptions().device(at::kCUDA, -1), false, false,
                  input.suggest_memory_format());  // 这里为异步操作, pytorch 自身cache pinned
                                                   // memory， 不怕析构
}

//  测试 cudaMemcpyAsync 是同步还是异步（非pinnedmemory）
// 结论 此时 cudaMemcpyAsync 和 cudaMemcpy 差不多
class TestRun {
 public:
  TestRun(){
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

at::Tensor async2cpu(at::Tensor input) {
  //  auto options=at::TensorOptions()
  //   .device(at::kCPU)
  //   .dtype(featRegion.dtype()) // at::kByte
  //   .layout(at::kStrided)
  //   .requires_grad(false).pinned_memory(false);
  at::TensorOptions options;
  if (input.device() == at::kCPU) {
    return input;
  } else
    options = at::TensorOptions().device(at::kCPU).pinned_memory(true);

  return input.to(options, true, false, at::MemoryFormat::Contiguous);  // 这里为异步操作
}

bool is_channel(at::Tensor in, unsigned right_index, unsigned wrong_index) {
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
at::Tensor tensor2nchw(at::Tensor in) {
  at::Tensor target;
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

at::Tensor img_1chw_guard(at::Tensor in) {
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

at::Tensor img_1hwc_guard(at::Tensor in) {
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

bool is_1chw(at::Tensor in) {
  const auto& in_size = in.sizes();

  if (in_size.size() == 4 && (in_size[1] == 1 || in_size[1] == 3 || in_size[1] == 4) &&
      in_size[0] == 1 && (in_size[2] * in_size[3] != 0)) {  // 1chw
    return true;
  }
  return false;
}

bool is_nchw(at::Tensor in) {
  const auto& in_size = in.sizes();

  if (in_size.size() == 4 && (in_size[1] == 1 || in_size[1] == 3 || in_size[1] == 4) &&
      (in_size[2] * in_size[3] != 0)) {  // nchw
    return true;
  }
  return false;
}

bool is_hwc(at::Tensor in) {
  const auto& in_size = in.sizes();

  if (in_size.size() == 3 && (in_size[2] == 1 || in_size[2] == 3 || in_size[2] == 4) &&
      (in_size[0] * in_size[1] != 0)) {  // hwc
    return true;
  }
  return false;
}

at::Tensor img_hwc_guard(at::Tensor in) {
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

at::Tensor tensor_permute(at::Tensor input, const std::vector<int>& min_shape,
                          const std::vector<int>& max_shape) {
  if (max_shape.size() != min_shape.size()) {
    throw std::runtime_error("max_shape.size() != min_shape.size()");
  }
  if (input.sizes().size() == max_shape.size() - 1) {
    input = input.unsqueeze(0);
  } else if (input.sizes().size() == max_shape.size()) {
    if (input.sizes()[0] > max_shape[0]) {
      throw std::runtime_error("data's batchsize should be smaller than max batch size");
    }
  } else {
    throw std::runtime_error("input data's dim not match model's. input = " +
                             std::to_string(input.sizes().size()));
  }
  bool need_permute = false;
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
      for (const auto& item : max_shape) ss << std::to_string(item) << ",";
      ss << "input data = ";
      for (const auto& item : input.sizes()) ss << std::to_string(item) << ",";
      ss << "please check it.";
      throw std::runtime_error("input shape and network shape not match. " + ss.str());
    }
    SPDLOG_DEBUG("implicit batch mode deprecated.");
    input = input.permute(permute_vector);
  }
  return input;
}

}  // namespace ipipe