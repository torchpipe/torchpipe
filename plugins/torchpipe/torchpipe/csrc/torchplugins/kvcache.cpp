#include <numeric>
#include <unordered_map>

#include "torchplugins/kvcache.hpp"
#include "hami/helper/macro.h"
#include "hami/helper/base_logging.hpp"

namespace torchpipe {

bool ReqPagePool::extend(const hami::id_type& name, size_t kv_num_tok) {
  auto iter = page_infos_.find(name);
  HAMI_ASSERT(iter != page_infos_.end());
  auto& info = iter->second;
  const auto total =
      (info.kv_page_indices.size() - 1) * page_size_ + info.kv_last_page_len;
  if (total >= kv_num_tok) {
    SPDLOG_WARN("decode_extend: total >= kv_num_tok");
    return true;
  } else if (page_size_ - info.kv_last_page_len >= kv_num_tok - total) {
    info.kv_last_page_len += kv_num_tok - total;
    return true;
  }
  //   PageInfo new_info;
  auto need_new_tok = kv_num_tok - (page_size_ - info.kv_last_page_len + total);
  auto kv_page_indices =
      page_pool_.alloc((need_new_tok + page_size_ - 1) / page_size_);
  if (kv_page_indices.empty()) {
    return false;
  }
  info.kv_page_indices.insert(
      info.kv_page_indices.end(),
      kv_page_indices.begin(),
      kv_page_indices.end());
  info.kv_last_page_len = (need_new_tok % page_size_) == 0
      ? need_new_tok
      : (need_new_tok % page_size_);
  return true;
}

bool ReqPagePool::alloc(const hami::id_type& name, size_t kv_num_tok) {
  std::unique_lock<std::mutex> lock(page_infos_lock_);
  if (page_infos_.size() >= max_num_req_) {
    return false;
  }

  HAMI_ASSERT(page_infos_.find(name) == page_infos_.end());

  auto [iter, inserted] = page_infos_.emplace(name, PageInfo{});
  if (!inserted)
    return false;
  auto& info = iter->second;

  info.kv_page_indices =
      page_pool_.alloc((kv_num_tok + page_size_ - 1) / page_size_);
  info.kv_last_page_len =
      (kv_num_tok % page_size_) == 0 ? kv_num_tok : (kv_num_tok % page_size_);

  if (info.kv_page_indices.empty()) {
    page_infos_.erase(name);
    return false;
  }

  return true;
}

} // namespace torchpipe
