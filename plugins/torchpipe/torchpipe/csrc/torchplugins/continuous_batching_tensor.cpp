#include "torchplugins/continuous_batching_tensor.hpp"
#include <torch/torch.h>

namespace torchpipe {

void TensorPage::forward(const omniback::dict& io) {
  auto table = page_table_->pop_activated();
  std::vector<int> all_indices;
  std::vector<int> all_lengths;

  for (const auto& item : table.first) {
    const omniback::PageTable::PageInfo& info = page_table_->page_info(item);

    // Append the page indices
    all_indices.insert(
        all_indices.end(),
        info.kv_page_indices.begin(),
        info.kv_page_indices.end());

    // Store the last page length
    all_lengths.push_back(info.kv_last_page_len);
  }

  // Convert to torch tensors
  auto options = torch::TensorOptions().dtype(torch::kInt32);
  torch::Tensor indices_tensor = torch::from_blob(
                                     all_indices.data(),
                                     {static_cast<int64_t>(all_indices.size())},
                                     options)
                                     .cuda();

  torch::Tensor lengths_tensor = torch::from_blob(
                                     all_lengths.data(),
                                     {static_cast<int64_t>(all_lengths.size())},
                                     options)
                                     .cuda();

  // Store in output dictionary
  (*io)["kv_page_indices"] = indices_tensor;
  (*io)["kv_last_page_len"] = lengths_tensor;
}
OMNI_REGISTER_BACKEND(TensorPage);
} // namespace torchpipe