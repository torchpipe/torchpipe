#include <numeric>
#include <unordered_map>

#include <cuda_runtime_api.h>
#include "c10/cuda/CUDAStream.h"
#include "omniback/builtin/page_table.hpp"
#include "omniback/core/backend.hpp"

#include <torch/torch.h>
#include "omniback/helper/base_logging.hpp"

namespace torchpipe {
class LocationManager {
 public:
};

using namespace omniback;
class FIAppendTensor : public omniback::BackendOne {
 private:
  size_t max_num_req_{16};
  size_t max_num_page_{0};
  size_t max_context_len_{4096};
  size_t num_layer_{32};
  size_t head_num_{32};
  size_t head_dim_{128};
  size_t page_size_{16};

 private:
  size_t max_num_page_per_seq_{0};
  bool inited_{false};
  // std::vector<torch::Tensor> k_;
  // std::vector<torch::Tensor> v_;
  std::unique_ptr<PageTable> pool_;

 private:
  void impl_init(
      const std::unordered_map<string, string>& params,
      const dict& options) override {
    str::try_update(params, "max_num_req", max_num_req_);
    max_num_page_ = str::get<size_t>(params, "max_num_page");
    // str::try_update(params, "max_context_len", max_context_len_);
    // str::try_update(params, "num_layer", num_layer_);
    // str::try_update(params, "head_num", head_num_);
    // str::try_update(params, "head_dim", head_dim_);
    OMNI_ASSERT(max_context_len_ % page_size_ == 0);
    max_num_page_per_seq_ = max_context_len_ / page_size_;
  }

  // void get(torch::Tensor kv_append_length) {
  //   for (size_t i = 0; i < kv_append_length.size(0); ++i) {
  //   }

  //   torch::Tensor batch_indices;
  //   torch::Tensor positions;

  //   torch::Tensor kv_page_indptr;
  //   torch::Tensor kv_last_page_len;
  //   torch::Tensor kv_append_indptr;
  // }
  torch::Tensor vec2tensor(const std::vector<int>& data) {
    thread_local auto options = torch::TensorOptions()
                                    .device(torch::kCUDA, -1)
                                    .dtype(torch::kInt) // torch::kByte
                                    .layout(torch::kStrided)
                                    .requires_grad(false);
    torch::Tensor re =
        torch::empty({static_cast<int64_t>(data.size())}, options);
    cudaError_t cuda_status = cudaMemcpyAsync(
        re.data_ptr(), // 目标设备指针
        data.data(), // 主机源指针
        data.size() * sizeof(int), // 字节大小
        cudaMemcpyHostToDevice, // 传输方向
        c10::cuda::getCurrentCUDAStream());
    if (cuda_status != cudaSuccess) {
      throw std::runtime_error(
          "CUDA 拷贝失败: " + std::string(cudaGetErrorString(cuda_status)));
    }
    return re;
  }

  void forward(const dict& io) override {
    if (!inited_)
      lazy_init();
    // in
    // id, type(prefill, decode), seq_len,
    // out
    bool success = true;
    std::vector<id_type> id = dict_gets<id_type>(io, "request_ids");
    // https://docs.flashinfer.ai/generated/flashinfer.page.append_paged_kv_cache.html#flashinfer.page.append_paged_kv_cache
    torch::Tensor seq_lens = dict_get<torch::Tensor>(io, "kv_append_length");
    OMNI_ASSERT(id.size() == seq_lens.size(0) && seq_lens.is_cpu());
    size_t total{0};

    for (size_t i = 0; i < id.size(); ++i) {
      SPDLOG_INFO("id = {}", id[i]);
      auto seq_len = seq_lens[i].item<int>();
      success = success && pool_->alloc(id[i], seq_len);
      OMNI_ASSERT(success);
      const auto& infor = pool_->page_info(id[i]);
      total += infor.kv_page_indices.size();
    }
    std::vector<int> kv_page_indices;
    kv_page_indices.reserve(total);

    std::vector<int> kv_page_indptr(1 + id.size(), 0);
    std::vector<int> kv_last_page_len(id.size());
    for (size_t i = 0; i < id.size(); ++i) {
      const auto& infor = pool_->page_info(id[i]);
      kv_page_indices.insert(
          kv_page_indices.end(),
          infor.kv_page_indices.begin(),
          infor.kv_page_indices.end());
      kv_page_indptr[i + 1] = kv_page_indptr[i] + infor.kv_page_indices.size();
      kv_last_page_len[i] = infor.kv_last_page_len;
    }

    auto kv_page_indices_CUDA = vec2tensor(kv_page_indices);
    auto kv_page_indptr_CUDA = vec2tensor(kv_page_indptr);
    auto kv_last_page_len_CUDA = vec2tensor(kv_last_page_len);

    (*io)["kv_page_indices"] = kv_page_indices_CUDA;
    (*io)["kv_page_indptr"] = kv_page_indptr_CUDA;
    (*io)["kv_last_page_len"] = kv_last_page_len_CUDA;
  }

  void lazy_init() {
    // if (max_num_page_ == 0) {
    //   auto stats = torch::cuda::memory_stats(-1);
    //   int64_t free_memory = stats.free_bytes; // 剩余显存字节数

    //   max_num_page_ = static_cast<size_t>(
    //       (free_memory * 0.9) /
    //       (page_size_ * head_num_ * head_dim_ * 2 /*kv*/ * num_layer_ *
    //        2 /*fp16 */));
    // }

    // k_.resize(num_layer_);
    // v_.resize(num_layer_);
    // auto options = torch::TensorOptions()
    //                    .device(torch::kCUDA, -1)
    //                    .dtype(torch::kFloat16)
    //                    .layout(torch::kStrided)
    //                    .requires_grad(false);
    // for (size_t layer_index = 0; layer_index < num_layer_; ++layer_index) {
    //   k_[layer_index] = torch::empty(
    //       {max_num_page_, page_size_, head_num_, head_dim_},
    //       options,
    //       torch::MemoryFormat::Contiguous);
    //   v_[layer_index] = torch::empty(
    //       {max_num_page_, page_size_, head_num_, head_dim_},
    //       options,
    //       torch::MemoryFormat::Contiguous);
    // }
    pool_ =
        std::make_unique<PageTable>(max_num_req_, max_num_page_, page_size_);
    inited_ = true;
  }
};
OMNI_REGISTER_BACKEND(FIAppendTensor);
} // namespace torchpipe
