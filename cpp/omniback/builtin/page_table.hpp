

#pragma once

#include <iterator>
#include <mutex>
#include <queue>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include "omniback/core/task_keys.hpp"

namespace omniback {
using omniback::id_type;

class ThreadSafeSlots {
 public:
  ThreadSafeSlots() = default;
  ThreadSafeSlots(size_t max_num) : max_num_(max_num) {
    std::lock_guard<std::mutex> lock(mutex_);
    free_slots_.reserve(max_num);
    std::generate_n(
        std::back_inserter(free_slots_), max_num, [n = 0]() mutable {
          return n++;
        });
  }
  void init(size_t max_num) {
    std::lock_guard<std::mutex> lock(mutex_);
    max_num_ = max_num;
    free_slots_.clear();
    free_slots_.reserve(max_num);
    std::generate_n(
        std::back_inserter(free_slots_), max_num, [n = 0]() mutable {
          return n++;
        });
  }

  void add_more_slots(size_t new_added_slots) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto ori = max_num_;
    max_num_ += new_added_slots;
    std::generate_n(
        std::back_inserter(free_slots_),
        new_added_slots,
        [ori, n = ori]() mutable { return n++; });
  }

  void free(int index) {
    std::lock_guard<std::mutex> lock(mutex_);
    free_slots_.push_back(index);
  }

  void free(const std::vector<int>& indices) {
    std::lock_guard<std::mutex> lock(mutex_);
    std::copy(indices.begin(), indices.end(), std::back_inserter(free_slots_));
  }

  std::vector<int> alloc(size_t need_size) {
    std::vector<int> allocated;
    allocated.reserve(need_size);

    std::lock_guard<std::mutex> lock(mutex_);
    if (free_slots_.size() < need_size)
      return {};

    auto begin = std::make_move_iterator(free_slots_.end() - need_size);
    auto end = std::make_move_iterator(free_slots_.end());
    allocated.assign(begin, end);
    free_slots_.erase(free_slots_.end() - need_size, free_slots_.end());

    return allocated;
  }

  size_t available_size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return free_slots_.size();
  }

 private:
  size_t max_num_{0};
  std::vector<int> free_slots_;
  mutable std::mutex mutex_;
};

class PageTable {
 public:
  struct PageInfo {
    int init_size{0};
    std::vector<int> kv_page_indices; // 页位置
    int kv_last_page_len = 0; // 最后一个页的长度
    float time{0}; // 最初时间
  };
  PageTable() = default;
  PageTable(size_t max_num_req, size_t max_num_page, size_t page_size)
      : max_num_req_(max_num_req),
        page_size_(page_size),
        max_num_page_(max_num_page),
        slots_(max_num_page) {
    page_infos_.reserve(max_num_req);
  }
  void init(size_t max_num_req, size_t max_num_page, size_t page_size) {
    max_num_req_ = (max_num_req);
    page_size_ = (page_size);
    max_num_page_ = max_num_page;
    slots_.init(max_num_page);
    page_infos_.reserve(max_num_req);
  }

  bool alloc(const omniback::id_type& name, size_t num_tok);
  // bool alloc_pages(const omniback::id_type& name, size_t num_page);
  bool alloc_or_reset(const omniback::id_type& name, size_t num_tok);

  bool reset(const omniback::id_type& name, size_t num_tok);
  bool extend(const omniback::id_type& name);

  bool free(const id_type& req) {
    std::lock_guard<std::mutex> lock(page_infos_lock_);
    auto iter = page_infos_.find(req);
    if (iter == page_infos_.end()) {
      return false;
    }
    slots_.free(iter->second.kv_page_indices);
    page_infos_.erase(iter);
    return true;
  }

  int get_num_tok(const id_type& id) const {
    std::lock_guard<std::mutex> lock(page_infos_lock_);
    const auto& item = page_infos_.at(id);
    if (item.kv_page_indices.empty())
      return 0;
    return item.kv_last_page_len +
        page_size_ * (item.kv_page_indices.size() - 1);
  }

  void add_more_page(size_t num_added_slots) {
    slots_.add_more_slots(num_added_slots);
  }

  size_t available_pages() {
    return slots_.available_size();
  }
  size_t available_ids() {
    std::lock_guard<std::mutex> lock(page_infos_lock_);
    return max_num_req_ - page_infos_.size();
  }

  void activate(std::vector<id_type> ids);
  void deactivate();

  size_t pop_all() {
    std::lock_guard<std::mutex> lock(page_infos_lock_);
    size_t final_size = ids_.size();
    std::queue<std::vector<id_type>> empty;
    std::swap(ids_, empty);
    return final_size;
  }
  std::pair<std::vector<id_type>, std::vector<int>> pop_activated();
  std::pair<std::vector<id_type>, std::vector<int>> get_activated();

  // std::vector<int> get_prefill_size(std::vector<id_type> ids);
  std::vector<int> get_prefill_size(const std::vector<id_type>& ids);

  std::vector<int> get_current_size(const std::vector<id_type>& ids);

  const PageInfo& page_info(const id_type& id) const {
    std::lock_guard<std::mutex> lock(page_infos_lock_);
    return page_infos_.at(id);
  }

  std::tuple<std::vector<int>, std::vector<int>, std::vector<int>> page_table(
      const std::vector<id_type>& id);

  int page_size() const {
    return page_size_;
  }
  int max_num_page() const {
    return max_num_page_;
  }

 private:
  float get_time();

 private:
  size_t max_num_req_{0};
  int page_size_{0};
  int max_num_page_{0};
  ThreadSafeSlots slots_;

  mutable std::mutex page_infos_lock_;
  std::unordered_map<id_type, PageInfo> page_infos_;
  std::queue<std::vector<id_type>> ids_;
};
PageTable& default_page_table(const std::string& tag = "");

} // namespace omniback
