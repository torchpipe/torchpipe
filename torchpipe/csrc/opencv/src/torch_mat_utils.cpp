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

#include "torch_mat_utils.hpp"
#include "torch_utils.hpp"

#include <opencv2/imgcodecs.hpp>
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

namespace ipipe {

static std::mutex __s_imwrite_mutex;  // protects imwrite with same file name

cv::Mat torchTensortoCVMat(torch::Tensor tensor, bool deepcopy) {  //
  tensor = img_hwc_guard(tensor);

  tensor = tensor.to(torch::kCPU, torch::kByte).contiguous();

  cv::Mat mat = cv::Mat(cv::Size(tensor.size(1), tensor.size(0)), CV_8UC(tensor.size(2)),
                        tensor.data_ptr<uchar>());
  assert(mat.isContinuous());
  if (deepcopy)
    return mat.clone();
  else
    return mat;
}

cv::Mat torchTensortoCVMatV2(torch::Tensor tensor, bool deepcopy) {  //
  tensor = img_hwc_guard(tensor);
  cv::Mat mat;
  tensor = tensor.to(torch::kCPU).contiguous();

  if (tensor.dtype() == torch::kByte) {
    mat = cv::Mat(cv::Size(tensor.size(1), tensor.size(0)), CV_8UC(tensor.size(2)),
                  tensor.data_ptr<uchar>());
  } else if (tensor.dtype() == torch::kFloat) {
    mat = cv::Mat(cv::Size(tensor.size(1), tensor.size(0)), CV_32FC(tensor.size(2)),
                  tensor.data_ptr<float>());
  } else if (tensor.dtype() == torch::kHalf) {
    tensor = tensor.to(torch::kFloat);
    mat = cv::Mat(cv::Size(tensor.size(1), tensor.size(0)), CV_32FC(tensor.size(2)),
                  tensor.data_ptr<float>());
  } else {
    throw std::runtime_error("unsupported datatype " + std::string(tensor.dtype().name()));
  }

  if (deepcopy) {
    mat = mat.clone();
    assert(mat.isContinuous());
    return mat;

  } else
    return mat;
}

void imwrite(std::string name, torch::Tensor tensor) {
  auto img = torchTensortoCVMat(tensor);
  std::lock_guard<std::mutex> lock(__s_imwrite_mutex);
  cv::imwrite(name, img);
}

void imwrite(std::string name, any tensor) {
  cv::Mat img = get_mat_from_any(tensor);
  std::lock_guard<std::mutex> lock(__s_imwrite_mutex);
  cv::imwrite(name, img);
}

// int leastPriority = -1, greatestPriority = -1;
// cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority);
// cudaStreamCreateWithPriority(&m_stream, cudaStreamNonBlocking,
//                              greatestPriority);getStreamFromExternal

#ifdef WITH_CUDA
torch::Tensor cvMat2TorchGPU(cv::Mat da, std::string data_format) {
  if (!da.isContinuous()) {
    da = da.clone();
  }
  const auto elesize = da.elemSize1();
  IPIPE_ASSERT(elesize == 1 || elesize == 4);

  auto image_tensor = torch::from_blob(da.data, {da.rows, da.cols, da.channels()},
                                       elesize == 1 ? torch::kByte : torch::kFloat);
  if (data_format == "nchw")
    image_tensor = image_tensor.cuda().permute({2, 0, 1}).unsqueeze(0);
  else
    image_tensor = image_tensor.cuda();
  return image_tensor;
}

torch::Tensor cvMat2TorchGPUV2(cv::Mat da) {
  if (!da.isContinuous()) {
    da = da.clone();
  }
  IPIPE_ASSERT(da.elemSize1() == 1 || da.elemSize1() == 4);
  auto options = torch::TensorOptions()
                     .device(torch::kCUDA, -1)
                     .dtype(da.elemSize1() == 1 ? torch::kByte : torch::kFloat)
                     .layout(torch::kStrided)
                     .requires_grad(false);
  auto image_tensor = torch::empty(
      {(signed long)1, (signed long)da.rows, (signed long)da.cols, (signed long)da.channels()},
      options, torch::MemoryFormat::Contiguous);
  auto* image = image_tensor.data_ptr<unsigned char>();
  cudaMemcpyAsync(image, da.data, da.rows * da.cols * da.elemSize(), cudaMemcpyHostToDevice,
                  c10::cuda::getCurrentCUDAStream());

  image_tensor = image_tensor.permute({0, 3, 1, 2});
  c10::cuda::getCurrentCUDAStream().synchronize();

  return image_tensor;
}
#endif

torch::Tensor cvMat2TorchCPU(cv::Mat da, bool deepcopy, std::string data_format) {  // todo deepcopy
  if (!da.isContinuous()) {
    da = da.clone();
  }

  auto image_tensor = torch::from_blob(da.data, {da.rows, da.cols, da.channels()},
                                       da.elemSize1() == 1 ? torch::kByte : torch::kFloat);
  if (data_format == "nchw") image_tensor = image_tensor.permute({2, 0, 1}).unsqueeze(0);

  if (deepcopy) {
    image_tensor = image_tensor.clone();
  }
  return image_tensor;
}
cv::Mat get_mat_from_any(any data) {
  cv::Mat out;
  if (data.type() == typeid(torch::Tensor)) {
    torch::Tensor tensor = any_cast<torch::Tensor>(data);
    out = torchTensortoCVMat(tensor);
  } else if (data.type() == typeid(std::vector<torch::Tensor>)) {
    torch::Tensor tensor = any_cast<std::vector<torch::Tensor>>(data)[0];
    out = torchTensortoCVMat(tensor);
  } else if (data.type() == typeid(cv::Mat)) {
    auto mat = any_cast<cv::Mat>(data);
    out = mat;
  } else if (data.type() == typeid(std::vector<cv::Mat>)) {
    auto mat = any_cast<std::vector<cv::Mat>>(data)[0];
    out = mat;
  }
  return out;
}

torch::Tensor get_tensor_from_any(any data) {
  torch::Tensor out;
  if (data.type() == typeid(torch::Tensor))
    out = any_cast<torch::Tensor>(data);
  else if (data.type() == typeid(std::vector<torch::Tensor>))
    out = any_cast<std::vector<torch::Tensor>>(data)[0];
  else if (data.type() == typeid(cv::Mat)) {
    auto mat = any_cast<cv::Mat>(data);
    out = cvMat2TorchCPU(mat);
  }
  return out;
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

  std::atomic<int> data_{1};
  std::mutex lock_;
};
// TestRun tmp;

}  // namespace ipipe
