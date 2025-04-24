
#include <fstream>

#include "helper/task_keys.hpp"
#include "helper/torch.hpp"
#include "torchplugins/GpuTensor.hpp"

using namespace hami;

namespace torchpipe {

/**
 * @brief cpu->gpu
 */

class GpuTensor : public hami::BackendOne {
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
HAMI_REGISTER(Backend, GpuTensor);

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

    if (input[TASK_DATA_KEY].type() == typeid(torch::Tensor)) {
      torch::Tensor input_tensor =
          any_cast<torch::Tensor>(input[TASK_DATA_KEY]);
      if (!input_tensor.is_cuda()) {
        SPDLOG_ERROR("input_tensor should be gpu tensor");
        throw std::runtime_error("input_tensor should be gpu tensor");
      }

      input[TASK_RESULT_KEY] = input_tensor.cpu();
    } else if (
        input[TASK_DATA_KEY].type() == typeid(std::vector<torch::Tensor>)) {
      std::vector<torch::Tensor> input_tensor =
          any_cast<std::vector<torch::Tensor>>(input[TASK_DATA_KEY]);
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
      SPDLOG_ERROR(
          "CpuTensor: torch::Tensor/std::vector<torch::Tensor> needed; "
          "error input type: " +
          std::string(input[TASK_DATA_KEY].type().name()));
      throw std::runtime_error(
          "CpuTensor: torch::Tensor/std::vector<torch::Tensor> needed; "
          "error input type: " +
          std::string(input[TASK_DATA_KEY].type().name()));
    }
  }
};

HAMI_REGISTER(hami::Backend, CpuTensor, "CpuTensor");

void IndexSelectTensor::impl_init(
    const std::unordered_map<std::string, std::string>& params,
    const dict& options) {
  parser_v2::ArgsKwargs args_kwargs =
      parser_v2::get_args_kwargs(this, default_cls_name(), params);
  HAMI_ASSERT(
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
    auto input = hami::dict_get<torch::Tensor>(io, TASK_DATA_KEY);
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
HAMI_REGISTER_BACKEND(IndexSelectTensor);

void EmbeddingTensor::impl_forward(const std::vector<dict>& ios) {
  for (const auto& io : ios) {
    auto input = hami::dict_get<torch::Tensor>(io, TASK_DATA_KEY);
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
HAMI_REGISTER_BACKEND(EmbeddingTensor);

class SetTensorRequestSize : public hami::BackendOne {
  void forward(const dict& io) override {
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
};

#if 0
class AppendIndexSelectTensor : public hami::Backend {
  void impl_init(
      const std::unordered_map<std::string, std::string>& params,
      const hami::dict& options) override {
    throw std::runtime_error("not impl");
    parser_v2::ArgsKwargs args_kwargs =
        parser_v2::get_args_kwargs(this, "AppendIndexSelectTensor", params);
    HAMI_ASSERT(
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
class PrintTensor : public hami::BackendOne {
  void impl_init(
      const std::unordered_map<std::string, std::string>& params,
      const hami::dict& options) override {}
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

HAMI_REGISTER_BACKEND(SetTensorRequestSize);
// HAMI_REGISTER_BACKEND(AppendIndexSelectTensor);
HAMI_REGISTER_BACKEND(PrintTensor);

} // namespace torchpipe