// #include <numeric>
// #include <unordered_map>

// #include "torchplugins/kvcache.hpp"
// #include "hami/core/backend.hpp"

// #include <torch/torch.h>

// namespace torchpipe {
// using namespace hami;
// class KVCacheTensor : public hami::BackendOne {
//  private:
//   size_t max_num_req_{16};
//   size_t max_num_page_{0};
//   size_t max_context_len_{4096};
//   size_t num_layer_{32};
//   size_t head_num_{32};
//   size_t head_dim_{128};
//   size_t page_size_{16};

//  private:
//   size_t max_num_page_per_seq_{0};
//   bool inited_{false};
//   std::vector<torch::Tensor> k_;
//   std::vector<torch::Tensor> v_;
//   std::unique_ptr<ReqPagePool> pool_;

//  private:
//   void impl_init(
//       const std::unordered_map<string, string>& params,
//       const dict& options) override {
//     str::try_update(params, "max_num_req", max_num_req_);
//     str::try_update(params, "max_num_page", max_num_page_);
//     str::try_update(params, "max_context_len", max_context_len_);
//     str::try_update(params, "num_layer", num_layer_);
//     str::try_update(params, "head_num", head_num_);
//     str::try_update(params, "head_dim", head_dim_);
//     HAMI_ASSERT(max_context_len_ % page_size_ == 0);
//     max_num_page_per_seq_ = max_context_len_ / page_size_;
//   }

//   void forward(const dict& io) override {
//     if (!inited_)
//       lazy_init();

//     throw std::runtime_error("forward(io) not supported by default");
//   }

//   void lazy_init() {
//     if (max_num_page_ == 0) {
//       auto stats = torch::cuda::memory_stats(-1);
//       int64_t free_memory = stats.free_bytes; // 剩余显存字节数

//       max_num_page_ = static_cast<size_t>(
//           (free_memory * 0.9) /
//           (page_size_ * head_num_ * head_dim_ * 2 /*kv*/ * num_layer_ *
//            2 /*fp16 */));
//     }

//     k_.resize(num_layer_);
//     v_.resize(num_layer_);
//     auto options = torch::TensorOptions()
//                        .device(torch::kCUDA, -1)
//                        .dtype(torch::kFloat16)
//                        .layout(torch::kStrided)
//                        .requires_grad(false);
//     for (size_t layer_index = 0; layer_index < num_layer_; ++layer_index) {
//       k_[layer_index] = torch::empty(
//           {max_num_page_, page_size_, head_num_, head_dim_},
//           options,
//           torch::MemoryFormat::Contiguous);
//       v_[layer_index] = torch::empty(
//           {max_num_page_, page_size_, head_num_, head_dim_},
//           options,
//           torch::MemoryFormat::Contiguous);
//     }
//     pool_ = std::make_unique<ReqPagePool>(max_num_req_, max_num_page_);
//     inited_ = true;
//   }
// };
// } // namespace torchpipe
