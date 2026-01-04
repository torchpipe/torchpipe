
#include <fstream>

#include "helper/task_keys.hpp"
#include "helper/torch.hpp"
#include "torchplugins/GpuTensor.hpp"
#include <tvm/ffi/error.h>

using namespace omniback;

namespace torchpipe {

/**
 * @brief cpu->gpu
 */

class GpuTensor : public omniback::BackendOne {
 public:
  void forward(const dict& io) override {
    auto data = dict_gets<torch::Tensor>(io, TASK_DATA_KEY);
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
OMNI_REGISTER(Backend, GpuTensor);

/**
 * @brief gpu->cpu
 */
class CpuTensor : public BackendOne {
 public:
  /**
   * @brief cpu->gpu
   * @param TASK_RESULT_KEY input[TASK_RESULT_KEY] =
   * input[TASK_DATA_KEY].cuda()
   */
  virtual void forward(const dict& io) override {
    auto& input = *io;

    if (auto opt = input[TASK_DATA_KEY].try_cast<torch::Tensor>()) {
      torch::Tensor input_tensor = opt.value();
      if (!input_tensor.is_cuda()) {
        SPDLOG_ERROR("input_tensor should be gpu tensor");
        throw std::runtime_error("input_tensor should be gpu tensor");
      }

      input[TASK_RESULT_KEY] = input_tensor.cpu();
    } else if (
        auto opt = input[TASK_DATA_KEY].try_cast<std::vector<torch::Tensor>>()) {
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
      }
    else {
      TVM_FFI_THROW(TypeError)<<(
          "CpuTensor: torch::Tensor/std::vector<torch::Tensor> needed; "
          "error input type: " );
    }
  }
};

OMNI_REGISTER(omniback::Backend, CpuTensor, "CpuTensor");

void IndexSelectTensor::impl_init(
    const std::unordered_map<std::string, std::string>& params,
    const dict& options) {
  parser_v2::ArgsKwargs args_kwargs =
      parser_v2::get_args_kwargs(this, reflect_cls_name(), params);
  OMNI_ASSERT(
      args_kwargs.first.size() == 1,
      "Requires exactly 1 argument. Usage: "
      "IndexSelectTensor(weight.pt)/IndexSelectTensor::args=weight.pt");
  std::string device = "cuda";
  str::try_update(args_kwargs.second, "device", device);

  const auto& name = args_kwargs.first.at(0);
  SPDLOG_INFO("load " + name);
  std::ifstream file(name);
  if (!file.good()) {
    throw std::invalid_argument(name + " not exists.");
  }
  file.seekg(0, file.end);
  int length = file.tellg();
  file.seekg(0, file.beg);

  std::vector<char> data(length);
  file.read(data.data(), length);

  device_ = torch::Device(device);

  weight_ = torch::pickle_load(data).toTensor().to(device_);
}

void IndexSelectTensor::impl_forward(const std::vector<dict>& ios) {
  for (const auto& io : ios) {
    auto input = omniback::dict_get<torch::Tensor>(io, TASK_DATA_KEY);
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

void EmbeddingTensor::impl_forward(const std::vector<dict>& ios) {
  for (const auto& io : ios) {
    auto input = omniback::dict_get<torch::Tensor>(io, TASK_DATA_KEY);
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

class SetTensorRequestSize : public omniback::Backend {
  void impl_forward(const std::vector<dict>& ios) override {
    for (const auto& io : ios) {
      auto data = dict_gets<torch::Tensor>(io, TASK_DATA_KEY);

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
class AppendIndexSelectTensor : public omniback::Backend {
  void impl_init(
      const std::unordered_map<std::string, std::string>& params,
      const omniback::dict& options) override {
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
class PrintTensor : public omniback::BackendOne {
  void impl_init(
      const std::unordered_map<std::string, std::string>& params,
      const omniback::dict& options) override {}
  void forward(const dict& io) override {
    auto data = dict_gets<torch::Tensor>(io, TASK_DATA_KEY);
    std::string id;
    try_update<std::string>(io, TASK_REQUEST_ID_KEY, id);
    std::ostringstream oss;
    for (size_t i = 0; i < data.size(); ++i) {
      oss << "Tensor " << i << " shape = " << data[i].sizes() << "\n";
    }

    for (const auto& item : data) {
      if (item.numel() > 60) {
        auto new_view = item.view(-1); // 将张量展平
        auto head = new_view.slice(0, 0, 5); // 取前5个元素
        auto tail = new_view.slice(0, -5, new_view.size(0)); // 取后5个元素
        oss << "Tensor is large. Shape: " << item.sizes()
            << ". Showing head and tail:\n";
        oss << "head: " << head << "\n...\ntail: " << tail << "\n";
      } else {
        oss << item << "\n\n";
      }
    }
    SPDLOG_WARN("PrintTensor({}): {}", id, oss.str());

    io->operator[](TASK_RESULT_KEY) = io->at(TASK_DATA_KEY);
  }
};

OMNI_REGISTER_BACKEND(SetTensorRequestSize);
// OMNI_REGISTER_BACKEND(AppendIndexSelectTensor);
OMNI_REGISTER_BACKEND(PrintTensor);

class OMNI_EXPORT LogGPUTime : public omniback::Backend {
 private:
  void impl_init(
      const std::unordered_map<std::string, std::string>& params,
      const dict& options) override final {
    auto args_kwargs =
        omniback::parser_v2::get_args_kwargs(this, "LogGPUTime", params);
    OMNI_ASSERT(
        args_kwargs.first.size() == 1,
        "Requires exactly ==1 argument. Usage: LogGPUTime(key)/LogGPUTime::args=key_to_time");
    key_ = args_kwargs.first[0];
  }
  void impl_forward(
      const std::vector<omniback::dict>& input_output) override final {
    // float time = get_time();
    c10::cuda::getCurrentCUDAStream().synchronize();
    float time = omniback::helper::timestamp();
    SPDLOG_INFO("timer: {} = {}", key_, time);
    for (const auto& item : input_output) {
      (*item)[TASK_RESULT_KEY] = item->at(TASK_DATA_KEY);
    }
  }
  [[nodiscard]] uint32_t impl_max() const override final {
    return max_;
  }

 private:
  float get_time() {
    return static_cast<float>(
        std::chrono::duration<double>(
            std::chrono::system_clock::now().time_since_epoch())
            .count());
  }
  size_t max_{std::numeric_limits<uint32_t>::max()};
  std::string key_;
};
OMNI_REGISTER_BACKEND(LogGPUTime);

} // namespace torchpipe