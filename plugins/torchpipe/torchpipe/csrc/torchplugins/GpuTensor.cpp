// Copyright 2021-2026 NetEase.
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

#include <fstream>
#include <sstream>

#include "helper/task_keys.hpp"
#include "helper/torch.hpp"
#include "torchplugins/GpuTensor.hpp"
#include <tvm/ffi/error.h>

namespace torchpipe {

/**
 * @brief Move tensor from CPU to GPU
 */
class GpuTensor : public om::BackendOne {
 public:
  void forward(const om::dict& io) override {
    auto data = om::dict_gets<torch::Tensor>(io, TASK_DATA_KEY);
    for (auto& item : data) {
      if (item.is_cpu()) {
        item = item.cuda();
      }
    }
    if (data.size() == 1)
      (*io)[TASK_RESULT_KEY] = data[0];
    else
      (*io)[TASK_RESULT_KEY] = data;
  }
};
OMNI_REGISTER(om::Backend, GpuTensor);

/**
 * @brief Move tensor from GPU to CPU
 */
class CpuTensor : public om::BackendOne {
 public:
  void forward(const om::dict& io) override {
    auto& input = *io;

    if (auto opt = input[TASK_DATA_KEY].try_cast<torch::Tensor>()) {
      torch::Tensor input_tensor = opt.value();
      if (!input_tensor.is_cuda()) {
        SPDLOG_ERROR("input_tensor should be gpu tensor");
        throw std::runtime_error("input_tensor should be gpu tensor");
      }
      input[TASK_RESULT_KEY] = input_tensor.cpu();
    } else if (auto opt = input[TASK_DATA_KEY].try_cast<std::vector<torch::Tensor>>()) {
      std::vector<torch::Tensor> input_tensor = opt.value();
      for (auto& item : input_tensor) {
        if (item.is_cuda()) {
          item = item.cpu();
        } else {
          SPDLOG_ERROR("input_tensor should be gpu tensor");
          throw std::runtime_error("input_tensor should be gpu tensor");
        }
      }
      input[TASK_RESULT_KEY] = input_tensor;
    } else {
      TVM_FFI_THROW(TypeError)
          << "CpuTensor: torch::Tensor or std::vector<torch::Tensor> required";
    }
  }
};

OMNI_REGISTER(om::Backend, CpuTensor, "CpuTensor");

void IndexSelectTensor::impl_init(
    const std::unordered_map<std::string, std::string>& params,
    const om::dict& options) {
  auto args_kwargs = om::parser_v2::get_args_kwargs(this, reflect_cls_name(), params);
  OMNI_ASSERT(args_kwargs.first.size() == 1,
              "Requires exactly 1 argument. Usage: IndexSelectTensor(weight.pt)");
  
  std::string device = "cuda";
  om::str::try_update(args_kwargs.second, "device", device);

  const auto& name = args_kwargs.first.at(0);
  SPDLOG_INFO("Loading weight from: {}", name);
  
  std::ifstream file(name, std::ios::binary);
  if (!file.good()) {
    throw std::invalid_argument(name + " does not exist.");
  }
  
  file.seekg(0, std::ios::end);
  const auto length = file.tellg();
  file.seekg(0, std::ios::beg);

  // Reserve capacity to avoid reallocations
  std::vector<char> data;
  data.reserve(static_cast<size_t>(length));
  data.resize(static_cast<size_t>(length));
  file.read(data.data(), length);

  device_ = torch::Device(device);
  weight_ = torch::pickle_load(data).toTensor().to(device_);
}

void IndexSelectTensor::impl_forward(const std::vector<om::dict>& ios) {
  for (const auto& io : ios) {
    auto input = om::dict_get<torch::Tensor>(io, TASK_DATA_KEY);
    if (device_ != input.device()) {
      input = input.to(device_);
    }
    if (input.sizes().size() == 2 && input.size(0) == 1) {
      input = input.squeeze(0);
    }

    torch::Tensor data_loaded = weight_.index_select(0, input);
    (*io)[TASK_RESULT_KEY] = data_loaded;
  }
}
OMNI_REGISTER_BACKEND(IndexSelectTensor);

void EmbeddingTensor::impl_forward(const std::vector<om::dict>& ios) {
  for (const auto& io : ios) {
    auto input = om::dict_get<torch::Tensor>(io, TASK_DATA_KEY);
    if (device_ != input.device()) {
      input = input.to(device_);
    }
    torch::Tensor data_loaded = torch::embedding(
        /*weight=*/weight_, // 加载的权重矩阵
        /*indices=*/input // .to(torch::kLong)
    );
    io->erase(TASK_DATA_KEY);
    (*io)[TASK_RESULT_KEY] = data_loaded;
  }
}
OMNI_REGISTER_BACKEND(EmbeddingTensor);

class SetTensorRequestSize : public om::Backend {
  void impl_forward(const std::vector<om::dict>& ios) override {
    for (const auto& io : ios) {
      auto data = om::dict_gets<torch::Tensor>(io, TASK_DATA_KEY);

      const size_t req_size = data.at(0).size(0);
      SPDLOG_DEBUG(
          "SetTensorRequestSize: req_size={}", req_size); // print_tensor(data),
      io->erase(TASK_DATA_KEY);
      (*io)[TASK_REQUEST_SIZE_KEY] = int(req_size);
      if (data.size() == 1)
        (*io)[TASK_RESULT_KEY] = data[0];
      else
        (*io)[TASK_RESULT_KEY] = data;
    }
  }
};

#if 0
class AppendIndexSelectTensor : public om::Backend {
  void impl_init(
      const std::unordered_map<std::string, std::string>& params,
      const om::dict& options) override {
    throw std::runtime_error("not impl");
    parser_v2::ArgsKwargs args_kwargs =
        parser_v2::get_args_kwargs(this, "AppendIndexSelectTensor", params);
    OMNI_ASSERT(
        args_kwargs.first.size() == 1,
        "Requires exactly 1 argument. Usage: AppendIndexSelectTensor(index)/AppendIndexSelectTensor::args=index");
    const auto& name = args_kwargs.first.at(0);

    SPDLOG_INFO("index = " + name);
    target_value_ = std::stoi(name);
    // cached_ = target_value_;
    tensor_cache_0_ = std::make_unique<torch::Tensor>(torch::tensor(
        {0}, torch::TensorOptions().dtype(torch::kLong).device("cuda")));
  }
  void impl_forward(const std::vector<dict>& ios) override {
    std::vector<int> req_sizes();
    req_sizes.reserve(ios.size());
    int sum = 0;
    for (const auto& io : ios) {
      sum += get_request_size(io);
      req_sizes.push_back(sum + target_value_);
    }
    static const auto opt =
        torch::TensorOptions().dtype(torch::kLong).device("cuda");
    torch::tensor(output_values, options);
  }
  // void forward(const std::vector<dict>& io) override {
  //   auto inputs = dict_gets<torch::Tensor>(io, TASK_DATA_KEY);
  //   IPIPE_ASSERT(!inputs.empty() && inputs[0].sizes().size() >= 2);

  //   const auto& input = inputs[0];
  //   int64_t index_select = input.size(-2);
  //   if (target_value_ < 0)
  //     index_select += target_value_;
  //   else {
  //     index_select = target_value_;
  //   }

  //   IPIPE_ASSERT(index_select >= 0 && index_select < input.size(-2));
  //   if (0 == index_select) {
  //     inputs.push_back(*tensor_cache_0_);
  //   } else {
  //     static const auto options =
  //         torch::TensorOptions().dtype(torch::kLong).device(torch::kCUDA);

  //     inputs.push_back(torch::tensor({index_select}, options));
  //   }

  //   (*io)[TASK_RESULT_KEY] = inputs;
  // }

 private:
  int target_value_{-1};
  // int cached_{-1};
  // std::unique_ptr<torch::Tensor> tensor_cache_;
  std::unique_ptr<torch::Tensor> tensor_cache_0_;
};
#endif
class PrintTensor : public om::BackendOne {
  void impl_init(
      const std::unordered_map<std::string, std::string>& params,
      const om::dict& options) override {}
  void forward(const om::dict& io) override {
    auto data = om::dict_gets<torch::Tensor>(io, TASK_DATA_KEY);
    std::string id;
    om::try_update<std::string>(io, om::TASK_REQUEST_ID_KEY, id);
    std::ostringstream oss;
    std::string result;
    result.reserve(data.size() * 100);

    for (size_t i = 0; i < data.size(); ++i) {
      oss << "Tensor " << i << " shape = " << data[i].sizes() << "\n";
    }

    for (const auto& item : data) {
      if (item.numel() > 60) {
        // Use const reference to avoid copy
        const auto new_view = item.view(-1);
        const auto head = new_view.slice(0, 0, 5);
        const auto tail = new_view.slice(0, -5, new_view.size(0));
        oss << "Tensor is large. Shape: " << item.sizes()
            << ". Showing head and tail:\n";
        oss << "head: " << head << "\n...\ntail: " << tail << "\n";
      } else {
        oss << item << "\n\n";
      }
    }
    SPDLOG_WARN("PrintTensor({}): {}", id, oss.str());

    // Use operator[] directly with at() for consistency
    (*io)[TASK_RESULT_KEY] = io->at(TASK_DATA_KEY);
  }
};

OMNI_REGISTER_BACKEND(SetTensorRequestSize);
// OMNI_REGISTER_BACKEND(AppendIndexSelectTensor);
OMNI_REGISTER_BACKEND(PrintTensor);

/**
 * @brief Log GPU timestamp for profiling
 */
class OMNI_EXPORT LogGPUTime : public om::Backend {
 private:
  size_t max_{std::numeric_limits<uint32_t>::max()};
  std::string key_;

  void impl_init(
      const std::unordered_map<std::string, std::string>& params,
      const om::dict& options) override final {
    auto args_kwargs = om::parser_v2::get_args_kwargs(this, "LogGPUTime", params);
    OMNI_ASSERT(args_kwargs.first.size() == 1,
                "Requires exactly 1 argument. Usage: LogGPUTime(key)");
    key_ = args_kwargs.first[0];
  }
  
  void impl_forward(const std::vector<om::dict>& input_output) override final {
    float time = om::helper::timestamp();
    SPDLOG_INFO("timer: {} = {}", key_, time);
    for (const auto& item : input_output) {
      (*item)[TASK_RESULT_KEY] = item->at(TASK_DATA_KEY);
    }
  }
  
  [[nodiscard]] uint32_t impl_max() const override final {
    return static_cast<uint32_t>(max_);
  }
};
OMNI_REGISTER_BACKEND(LogGPUTime);

} // namespace torchpipe