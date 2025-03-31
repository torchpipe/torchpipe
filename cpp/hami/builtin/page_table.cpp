#include <numeric>
#include <unordered_map>
#include <shared_mutex>
// #include <shared_lock>

#include "hami/builtin/page_table.hpp"
#include "hami/helper/macro.h"
#include "hami/helper/base_logging.hpp"

namespace hami {

bool PageTable::extend(const hami::id_type& name, size_t num_tok) {
  auto iter = page_infos_.find(name);
  HAMI_ASSERT(iter != page_infos_.end());
  auto& info = iter->second;
  const auto total =
      (info.kv_page_indices.size() - 1) * page_size_ + info.kv_last_page_len;
  if (total >= num_tok) {
    SPDLOG_WARN("decode_extend: total >= num_tok");
    return true;
  } else if (page_size_ - info.kv_last_page_len >= num_tok - total) {
    info.kv_last_page_len += num_tok - total;
    return true;
  }
  //   PageInfo new_info;
  auto need_new_tok = num_tok - (page_size_ - info.kv_last_page_len + total);
  auto kv_page_indices =
      page_table_.alloc((need_new_tok + page_size_ - 1) / page_size_);
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

void PageTable::activate(std::vector<id_type> ids) {
  std::lock_guard<std::mutex> lock(page_infos_lock_);
  for (const auto& item : ids) {
    HAMI_ASSERT(page_infos_.find(item) != page_infos_.end());
  }
  ids_.push(std::move(ids));
}

bool PageTable::alloc(const hami::id_type& name, size_t num_tok) {
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
      page_table_.alloc((num_tok + page_size_ - 1) / page_size_);
  info.kv_last_page_len =
      (num_tok % page_size_) == 0 ? num_tok : (num_tok % page_size_);

  if (info.kv_page_indices.empty()) {
    page_infos_.erase(name);
    return false;
  }

  return true;
}

PageTable& default_page_table(const std::string& tag) {
  static std::shared_mutex map_mutex;
  static std::unordered_map<std::string, std::shared_ptr<PageTable>>
      page_table_map;

  {
    std::shared_lock lock(map_mutex);
    auto it = page_table_map.find(tag);
    if (it != page_table_map.end()) {
      return *(it->second);
    }
  }

  std::unique_lock lock(map_mutex);
  auto& ptr = page_table_map[tag];
  if (!ptr) {
    ptr = std::make_shared<PageTable>();
  }
  return *ptr;
}

} // namespace hami
