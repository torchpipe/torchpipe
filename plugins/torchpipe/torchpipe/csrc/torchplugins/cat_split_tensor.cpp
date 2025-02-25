#include "torchplugins/cat_split_tensor.hpp"
#include "helper/task_keys.hpp"
#include "helper/torch.hpp"

using namespace hami;

namespace torchpipe {

void CatTensor::init(
    const std::unordered_map<std::string, std::string>& config_param,
    const dict& dict_config) {
    if (dict_config &&
        dict_config->find(TASK_IO_INFO_KEY) != dict_config->end()) {
        net_shapes_ = dict_get<std::shared_ptr<NetIOInfos>>(dict_config,
                                                            TASK_IO_INFO_KEY);
        for (const auto& item : net_shapes_->first) {
            HAMI_ASSERT(item.max.nbDims == item.min.nbDims &&
                        0 != item.max.nbDims);
        }
    }
}

void CatTensor::forward(const std::vector<dict>& input_dict) {
    std::vector<std::vector<torch::Tensor>> cated_inputs;
    std::vector<size_t> req_size;
    for (const auto& input : input_dict) {
        auto data = dict_gets<torch::Tensor>(input, TASK_DATA_KEY);
        if (net_shapes_) {
            fix_tensors(data, net_shapes_);
        }
        req_size.push_back(data[0].size(0));
        cated_inputs.push_back(std::move(data));
    }
    req_size[0] = std::accumulate(req_size.begin(), req_size.end(), 0ULL);

    // row2col
    const size_t num_tensors = cated_inputs.front().size();
    // const size_t batch_size = cated_inputs.size();
    std::vector<std::vector<torch::Tensor>> nchws(num_tensors);
    for (size_t i = 0; i < cated_inputs.size(); ++i) {
        HAMI_ASSERT(cated_inputs[i].size() == num_tensors,
                    "All inputs must have the same number of tensors");
        for (size_t j = 0; j < num_tensors; ++j) {
            nchws[j].push_back(cated_inputs[i][j]);
        }
    }

    // cat for result
    std::vector<torch::Tensor> result;

    for (const auto& item : nchws) {
        if (item.size() == 1)
            result.push_back(item[0]);
        else
            result.push_back(torch::cat(item, 0));
    }
    if (result.size() == 1) {
        (*input_dict.front())[TASK_RESULT_KEY] = result[0];
    } else {
        (*input_dict.front())[TASK_RESULT_KEY] = result;
    }

    for (size_t i = 0; i < input_dict.size(); ++i) {
        (*input_dict[i])[TASK_REQUEST_SIZE_KEY] = req_size[i];
    }
}

HAMI_REGISTER(Backend, CatTensor);

class ContiguousTensor : public hami::SingleBackend {
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
HAMI_REGISTER(Backend, ContiguousTensor);

class GpuTensor : public hami::SingleBackend {
   public:
    void forward(const dict& input_output) override {
        auto data = dict_gets<torch::Tensor>(input_output, TASK_DATA_KEY);
        for (auto& item : data) {
            if (item.is_cpu()) {
                item = item.cuda();
            }
        }
        if (data.size() == 1)
            (*input_output)[TASK_RESULT_KEY] = data[0];
        else
            (*input_output)[TASK_RESULT_KEY] = data;
    }
};
HAMI_REGISTER(Backend, GpuTensor);

void SplitTensor::init(
    const std::unordered_map<std::string, std::string>& config_param,
    const dict& dict_config) {}
void SplitTensor::forward(const std::vector<dict>& input_dict) {
    std::vector<torch::Tensor> cated_inputs =
        dict_gets<torch::Tensor>(input_dict.front(), TASK_DATA_KEY);

    size_t curr_index = cated_inputs[0].size(0);

    std::vector<std::vector<torch::Tensor>> results(input_dict.size());
    for (auto index = input_dict.size() - 1; index > 1; --index) {
        const size_t req_size =
            dict_get<int>(input_dict[index], TASK_REQUEST_SIZE_KEY);
        curr_index -= req_size;
        for (const auto& item : cated_inputs) {
            results[index].push_back(item.index(
                {torch::indexing::Slice(curr_index, curr_index + req_size)}));
        }
    }

    for (const auto& item : cated_inputs) {
        results[0].push_back(
            item.index({torch::indexing::Slice(0, curr_index)}));
    }

    (*input_dict[0])[TASK_REQUEST_SIZE_KEY] = curr_index;
    for (size_t i = 0; i < input_dict.size(); ++i) {
        if (results[i].size() == 1)
            (*input_dict[i])[TASK_RESULT_KEY] = results[i][0];
        else
            (*input_dict[i])[TASK_RESULT_KEY] = results[i];
    }
}
HAMI_REGISTER(Backend, SplitTensor);

}  // namespace torchpipe