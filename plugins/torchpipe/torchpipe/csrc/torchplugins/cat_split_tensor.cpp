#include "torchplugins/cat_split_tensor.hpp"
#include "helper/task_keys.hpp"
#include "helper/torch.hpp"
#include "omniback/helper/timer.hpp"

using namespace omniback;

namespace torchpipe {
void CatTensor::impl_init(
    const std::unordered_map<std::string, std::string>& params,
    const dict& kwargs) {
  auto iter = params.find("append_index_selector");
  if (iter != params.end()) {
    index_selector_ = stoi(iter->second);
    SPDLOG_INFO("index_selector = {}", *index_selector_);
  }
}

void CatTensor::impl_forward(const std::vector<dict>& input_dict) {
  std::vector<std::vector<torch::Tensor>> cated_inputs;
  std::vector<int> req_size;
  for (const auto& input : input_dict) {
    auto data = dict_gets<torch::Tensor>(input, TASK_DATA_KEY);

    req_size.push_back(data[0].size(0));
    cated_inputs.push_back(std::move(data));
  }
  auto total_bs = std::accumulate(req_size.begin(), req_size.end(), 0);

  // row2col
  const size_t num_tensors = cated_inputs.front().size();

  // const size_t batch_size = cated_inputs.size();
  std::vector<std::vector<torch::Tensor>> nchws(num_tensors);
  for (size_t i = 0; i < cated_inputs.size(); ++i) {
    OMNI_ASSERT(
        cated_inputs[i].size() == num_tensors,
        "All inputs must have the same number of tensors");
    for (size_t j = 0; j < num_tensors; ++j) {
      nchws[j].push_back(cated_inputs[i][j]);
    }
  }

  // cat for result
  std::vector<torch::Tensor> result;

  for (const auto& item : nchws) {
    if (item.size() == 1)
      result.push_back(item.at(0));
    else
      result.push_back(torch::cat(item, 0));
  }
  if (index_selector_) {
    std::vector<int64_t> output_values;
    output_values.reserve(req_size.size());
    int current_sum = 0;
    for (int size : req_size) {
      current_sum += size;
      output_values.push_back(current_sum + *index_selector_);
    }

    const auto opt =
        torch::TensorOptions().dtype(torch::kLong).device(torch::kCUDA);
    result.push_back(torch::tensor(output_values, opt));
    // std::ostringstream oss;
    // for (size_t i = 0; i < result.size(); ++i) {
    //   oss << "i="<< i<<": ";
    //   oss << result[i].sizes() << ",";
    // }
    // SPDLOG_DEBUG("CatTensor output sizes: {}", oss.str());
    (*input_dict.front())[TASK_RESULT_KEY] = result;
  } else {
    if (result.size() == 1) {
      (*input_dict.front())[TASK_RESULT_KEY] = result[0];
    } else {
      (*input_dict.front())[TASK_RESULT_KEY] = result;
    }
  }

  (*input_dict[0])[TASK_REQUEST_SIZE_KEY] = total_bs;
  for (size_t i = 1; i < input_dict.size(); ++i) {
    (*input_dict[i])[TASK_REQUEST_SIZE_KEY] = int(req_size[i]);
  }
}

OMNI_REGISTER(Backend, CatTensor);

void FixTensor::impl_init(
    const std::unordered_map<std::string, std::string>& config_param,
    const dict& kwargs) {
  if (kwargs && kwargs->find(TASK_IO_INFO_KEY) != kwargs->end()) {
    net_shapes_ =
        dict_get<std::shared_ptr<NetIOInfos>>(kwargs, TASK_IO_INFO_KEY);
    for (const auto& item : net_shapes_->first) {
      OMNI_ASSERT(item.max.nbDims == item.min.nbDims && 0 != item.max.nbDims);
    }
  }
}

void FixTensor::impl_forward(const std::vector<dict>& input_dict) {
  for (const auto& input : input_dict) {
    auto data = dict_gets<torch::Tensor>(input, TASK_DATA_KEY);
    if (net_shapes_) {
      fix_tensors(data, net_shapes_);
    }

    if (data.size() == 1) {
      (*input)[TASK_RESULT_KEY] = data[0];
    } else {
      (*input)[TASK_RESULT_KEY] = data;
    }
  }
}

OMNI_REGISTER(Backend, FixTensor);

class ContiguousTensor : public omniback::BackendOne {
 public:
  void forward(const dict& input_output) override {
    auto data = dict_gets<torch::Tensor>(input_output, TASK_DATA_KEY);
    for (auto& item : data) {
      if (!item.is_contiguous()) {
        item = item.contiguous();
      }
    }
    if (data.size() == 1)
      (*input_output)[TASK_RESULT_KEY] = data[0];
    else
      (*input_output)[TASK_RESULT_KEY] = data;
  }
};
OMNI_REGISTER(Backend, ContiguousTensor);

void SplitTensor::impl_init(
    const std::unordered_map<std::string, std::string>& config_param,
    const dict& kwargs) {}
void SplitTensor::impl_forward(const std::vector<dict>& input_dict) {
  std::vector<torch::Tensor> cated_inputs =
      dict_gets<torch::Tensor>(input_dict.front(), TASK_DATA_KEY);
  std::vector<int> req_sizes(input_dict.size(), 1);
  size_t curr_index = cated_inputs[0].size(0);
  if (curr_index == input_dict.size()) {
    // OMNI_FATAL_ASSERT(
    //     cated_inputs.size() == 1,
    //     "no implementation for multiple outputs yet"); // todo
    // for (auto i = 0; i < curr_index; ++i) {
    //   (*input_dict[i])[TASK_RESULT_KEY] = cated_inputs[0][i];
    //   (*input_dict[i])[TASK_REQUEST_SIZE_KEY] = 1;
    // }
    // return;
  } else {
    for (auto i = 0; i < input_dict.size(); ++i) {
      req_sizes[i] = get_request_size(input_dict[i]);
    }
  }

  std::vector<std::vector<torch::Tensor>> results(input_dict.size());
  for (auto index = input_dict.size() - 1; index >= 1; --index) {
    // const size_t req_size =
    //     dict_get<int>(input_dict[index], TASK_REQUEST_SIZE_KEY);
    const size_t req_size = req_sizes[index];
    // SPDLOG_INFO("req_size = {}/{}", req_size, input_dict.size());
    OMNI_FATAL_ASSERT(
        curr_index > req_size,
        "curr_index=" + std::to_string(curr_index) +
            ", req_size=" + std::to_string(req_size));
    curr_index -= req_size;
    for (const auto& item : cated_inputs) {
      results[index].push_back(item.index(
          {torch::indexing::Slice(curr_index, curr_index + req_size)}));
    }
  }

  for (const auto& item : cated_inputs) {
    results[0].push_back(item.index({torch::indexing::Slice(0, curr_index)}));
  }

  (*input_dict[0])[TASK_REQUEST_SIZE_KEY] = int(curr_index);
  for (size_t i = 0; i < input_dict.size(); ++i) {
    if (results[i].size() == 1)
      (*input_dict[i])[TASK_RESULT_KEY] = results[i][0];
    else
      (*input_dict[i])[TASK_RESULT_KEY] = results[i];
  }
}
OMNI_REGISTER(Backend, SplitTensor);

void ArgMaxTensor::impl_init(
    const std::unordered_map<std::string, std::string>& params,
    const dict& options) {}
void ArgMaxTensor::impl_forward(const std::vector<dict>& io) {
  for (const auto& item : io) {
    auto input_tensor = dict_get<torch::Tensor>(item, TASK_DATA_KEY);

    // IPIPE_ASSERT(input_tensor.sizes().size() == 2);
    // torch::Tensor output = input_tensor.softmax(-1);
    auto max_index = torch::argmax(input_tensor, -1);

    (*item)[TASK_RESULT_KEY] = max_index;
  }
}

void SoftmaxArgMaxTensor::impl_forward(const std::vector<dict>& ios) {
  for (const auto& item : ios) {
    auto input_tensor = dict_get<torch::Tensor>(item, TASK_DATA_KEY);

    input_tensor = torch::softmax(input_tensor, -1);
    auto max_index = torch::argmax(input_tensor, -1);

    (*item)[TASK_RESULT_KEY] = max_index;
  }
}

OMNI_REGISTER_BACKEND(SoftmaxArgMaxTensor);

OMNI_REGISTER_BACKEND(ArgMaxTensor);
} // namespace torchpipe