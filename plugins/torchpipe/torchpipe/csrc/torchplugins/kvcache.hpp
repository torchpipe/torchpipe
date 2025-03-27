

#pragma once

#include <mutex>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <iterator>
#include "hami/core/task_keys.hpp"

namespace torchpipe {
using hami::id_type;

class ThreadSafeSlots {
 public:
  ThreadSafeSlots(size_t max_num) : max_num_(max_num) {
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

class ReqPagePool {
 public:
  struct PageInfo {
    std::vector<int> kv_page_indices; // 页位置
    int kv_last_page_len = 0; // 最后一个页的长度
  };
  ReqPagePool(size_t max_num_req, size_t max_num_page, size_t page_size)
      : max_num_req_(max_num_req),
        page_size_(page_size),
        page_pool_(max_num_page) {
    page_infos_.reserve(max_num_req);
  }

  bool alloc(const hami::id_type& name, size_t kv_num_tok);

  bool extend(const hami::id_type& name, size_t kv_num_tok);

  void free(const id_type& req) {
    std::lock_guard<std::mutex> lock(page_infos_lock_);
    page_pool_.free(page_infos_.at(req).kv_page_indices);
    page_infos_.erase(req);
  }

  void add_more_page(size_t num_added_slots) {
    page_pool_.add_more_slots(num_added_slots);
  }

  size_t available_pages() {
    return page_pool_.available_size();
  }
  size_t available_ids() {
    std::lock_guard<std::mutex> lock(page_infos_lock_);
    return max_num_req_ - page_infos_.size();
  }

  const PageInfo& page_info(const id_type& id) const {
    return page_infos_.at(id);
  }

 private:
  size_t max_num_req_{0};
  int page_size_{0};
  ThreadSafeSlots page_pool_;

  std::mutex page_infos_lock_;
  std::unordered_map<id_type, PageInfo> page_infos_;
};
} // namespace torchpipe
