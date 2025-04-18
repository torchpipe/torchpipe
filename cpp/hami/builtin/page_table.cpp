#include <numeric>
#include <unordered_map>
#include <shared_mutex>
// #include <shared_lock>

#include "hami/builtin/page_table.hpp"
#include "hami/helper/macro.h"
#include "hami/helper/base_logging.hpp"
#include "hami/helper/timer.hpp"

namespace hami {

bool PageTable::reset(const hami::id_type& name, size_t num_tok) {
  // std::lock_guard<std::mutex> lock(page_infos_lock_);
  // auto iter = page_infos_.find(name);
  // HAMI_ASSERT(iter != page_infos_.end());
  // auto& info = iter->second;
  const auto total = get_num_tok(name);
  if (total >= num_tok) {
    SPDLOG_WARN("extend: total >= num_tok");
    return true;
  }
  SPDLOG_INFO(
      "PageTable::reset(tokens): id={},now={},required={}",
      name,
      total,
      num_tok);

  std::lock_guard<std::mutex> lock(page_infos_lock_);
  auto& info = page_infos_.at(name);

  if (page_size_ - info.kv_last_page_len >= num_tok - total) {
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

float PageTable::get_time() {
  static const auto start_time = helper::now();
  return helper::time_passed(start_time);
}

bool PageTable::extend(const hami::id_type& name) {
  // std::lock_guard<std::mutex> lock(page_infos_lock_);
  // auto iter = page_infos_.find(name);
  // HAMI_ASSERT(iter != page_infos_.end());
  // auto& info = iter->second;
  // const auto total = get_num_tok(name);

  std::lock_guard<std::mutex> lock(page_infos_lock_);
  auto& info = page_infos_.at(name);

  if (page_size_ - info.kv_last_page_len >= 1) {
    info.kv_last_page_len += 1;
    return true;
  }
  //   PageInfo new_info;
  auto kv_page_indices = page_table_.alloc(1);
  if (kv_page_indices.empty()) {
    return false;
  }
  info.kv_page_indices.insert(
      info.kv_page_indices.end(),
      kv_page_indices.begin(),
      kv_page_indices.end());
  info.kv_last_page_len = 1;
  return true;
}

std::pair<std::vector<id_type>, std::vector<int>> PageTable::pop_activated() {
  std::pair<std::vector<id_type>, std::vector<int>> re;

  std::lock_guard<std::mutex> lock(page_infos_lock_);
  if (ids_.empty())
    return re;
  // re.first = ids_.front();
  std::swap(ids_.front(), re.first);
  ids_.pop();
  for (const id_type& id : re.first) {
    auto iter = page_infos_.find(id);
    HAMI_FATAL_ASSERT(
        iter != page_infos_.end(),
        id + " not found. Size = " + std::to_string(page_infos_.size()) +
            " name = " + page_infos_.begin()->first);
    const auto& item = iter->second;
    HAMI_FATAL_ASSERT(!item.kv_page_indices.empty());

    re.second.push_back(
        item.kv_last_page_len + page_size_ * (item.kv_page_indices.size() - 1));
  }

  return re;
}
std::vector<int> PageTable::get_prefill_num_req_toks(
    const std::vector<id_type>& ids) {
  std::vector<int> re;
  std::lock_guard<std::mutex> lock(page_infos_lock_);

  for (const id_type& id : ids) {
    auto iter = page_infos_.find(id);
    HAMI_FATAL_ASSERT(
        iter != page_infos_.end(),
        id + " not found. Size = " + std::to_string(page_infos_.size()) +
            " name = " + page_infos_.begin()->first);
    const auto& item = iter->second;

    re.push_back(item.init_size);
  }

  return re;
}

std::pair<std::vector<id_type>, std::vector<int>> PageTable::get_activated() {
  std::pair<std::vector<id_type>, std::vector<int>> re;

  std::lock_guard<std::mutex> lock(page_infos_lock_);
  if (ids_.empty())
    return re;
  re.first = ids_.front();
  // std::swap(ids_.front(), re.first);
  // ids_.pop();
  for (const id_type& id : re.first) {
    auto iter = page_infos_.find(id);
    HAMI_FATAL_ASSERT(
        iter != page_infos_.end(),
        id + " not found. Size = " + std::to_string(page_infos_.size()) +
            " name = " + page_infos_.begin()->first);
    const auto& item = iter->second;
    HAMI_FATAL_ASSERT(!item.kv_page_indices.empty());

    re.second.push_back(
        item.kv_last_page_len + page_size_ * (item.kv_page_indices.size() - 1));
  }

  return re;
}

std::tuple<std::vector<int>, std::vector<int>, std::vector<int>> PageTable::
    page_table(const std::vector<id_type>& id) {
  // std::vector<int> kv_page_indices;
  // std::vector<int> kv_page_indptr;
  // std::vector<int> kv_last_page_len;

  // HAMI_ASSERT(id.size() == seq_lens.size(0) && seq_lens.is_cpu());
  size_t total{0};

  std::lock_guard<std::mutex> lock(page_infos_lock_);
  for (size_t i = 0; i < id.size(); ++i) {
    total += page_infos_.at(id[i]).kv_page_indices.size();
  }

  std::vector<int> kv_page_indices;
  kv_page_indices.reserve(total);

  std::vector<int> kv_page_indptr(1 + id.size(), 0);
  std::vector<int> kv_last_page_len(id.size());
  for (size_t i = 0; i < id.size(); ++i) {
    const auto& infor = page_infos_.at(id[i]);
    kv_page_indices.insert(
        kv_page_indices.end(),
        infor.kv_page_indices.begin(),
        infor.kv_page_indices.end());
    kv_page_indptr[i + 1] = kv_page_indptr[i] + infor.kv_page_indices.size();
    kv_last_page_len[i] = infor.kv_last_page_len;
  }

  return std::make_tuple(kv_page_indices, kv_page_indptr, kv_last_page_len);
}

void PageTable::activate(std::vector<id_type> ids) {
  std::lock_guard<std::mutex> lock(page_infos_lock_);
  for (const auto& item : ids) {
    HAMI_ASSERT(page_infos_.find(item) != page_infos_.end(), "no" + item);
  }
  ids_.push(std::move(ids));
}

void PageTable::deactivate() {
  std::lock_guard<std::mutex> lock(page_infos_lock_);
  ids_ = {};
  // decltype(ids_) empty;
  // std::swap(ids_, empty);
}

// bool PageTable::alloc_pages(const hami::id_type& name, size_t num_page) {
//   std::unique_lock<std::mutex> lock(page_infos_lock_);

//   auto& info = page_infos_.at(name);

//   if (page_size_ - info.kv_last_page_len >= num_tok - total) {
//     info.kv_last_page_len += num_tok - total;
//     return true;
//   }
//   //   PageInfo new_info;
//   auto need_new_tok = num_tok - (page_size_ - info.kv_last_page_len + total);
//   auto kv_page_indices =
//       page_table_.alloc((need_new_tok + page_size_ - 1) / page_size_);
//   if (kv_page_indices.empty()) {
//     return false;
//   }
//   info.kv_page_indices.insert(
//       info.kv_page_indices.end(),
//       kv_page_indices.begin(),
//       kv_page_indices.end());
//   info.kv_last_page_len = (need_new_tok % page_size_) == 0
//       ? need_new_tok
//       : (need_new_tok % page_size_);
//   return true;
// }

bool PageTable::alloc_or_reset(const hami::id_type& name, size_t num_tok) {
  bool find_id = false;
  {
    std::unique_lock<std::mutex> lock(page_infos_lock_);
    find_id = page_infos_.find(name) != page_infos_.end();
  }
  // todo: lock
  if (find_id)
    return reset(name, num_tok);
  else
    return alloc(name, num_tok);
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
  info.init_size = num_tok;
  info.time = get_time();
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
