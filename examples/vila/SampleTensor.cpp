#include "Backend.hpp"
#include "params.hpp"
#include <torch/torch.h>
namespace ipipe {
class SampleTensor : public SingleBackend {
 public:
  virtual bool init(const std::unordered_map<std::string, std::string> &config_param,
                    dict) override {
    params_ = std::unique_ptr<Params>(
        new Params({{"top_k", "50"}, {"top_p", "0.2"}}, {"temperature"}, {}, {}));
    if (!params_->init(config_param)) return false;
    temperature_ = std::stof(params_->at("temperature"));
    topk_ = std::stoi(params_->at("top_k"));
    topp_ = std::stof(params_->at("top_p"));
    return true;
  }
  virtual void forward(dict input_dict) override {
    auto &input = *input_dict;
    torch::Tensor input_tensor = any_cast<torch::Tensor>(input[TASK_DATA_KEY]);
    // std::cout << "start " << input_tensor << std::endl;
    // input_tensor = input_tensor.index({"...", -1, "..."});
    // no extra batch dim
    input_tensor = input_tensor.index({-1});
    // std::cout << input_tensor << std::endl;
    auto scores = input_tensor / temperature_;

    if (topk_ > 0) {
      auto indices_to_remove =
          scores < std::get<0>(torch::topk(scores, topk_)).index({-1, torch::indexing::None});
      scores.masked_fill_(indices_to_remove, -std::numeric_limits<float>::infinity());
    }

    auto probs = torch::softmax(scores, /*dim=*/-1);

    auto next_tokens = torch::multinomial(probs, /*num_samples=*/1);
    // next_tokens = next_tokens - next_tokens + 1;
    input[TASK_RESULT_KEY] = next_tokens;
    input["tensor_item"] = next_tokens.item<long>();
  }

 private:
  std::unique_ptr<Params> params_;
  float temperature_;
  int topk_;
  float topp_;
};

IPIPE_REGISTER(Backend, SampleTensor, "SampleTensor")
};  // namespace ipipe
