

#pragma once

#include <vector>
#include <string>
#include <mutex>
#include <set>
#include <unordered_set>
#include <unordered_map>

namespace torchpipe {
class ThreadSafeSlots {
 public:
  ThreadSafeSlots(size_t max_num) : max_num_(max_num) {
    free_slots_.reserve(max_num);
    for (int i = 0; i < max_num; ++i) {
      free_slots_.insert(i);
    }
  }

  void add_more_slots(size_t new_added_slots) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto ori = max_num_;
    max_num_ += new_added_slots;
    for (int i = ori; i < max_num_; ++i) {
      free_slots_.insert(i);
    }
  }

  bool free(int index) {
    if (index < 0 || index >= max_num_)
      return false;

    std::lock_guard<std::mutex> lock(mutex_);
    free_slots_.insert(index);
    return true;
  }

  void free(std::string name) {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto& e_data = id_.at(name);
    for (const auto& item : e_data)
      free_slots_.erase(item);
    id_.erase(name);
  }

  bool alloc(const std::vector<std::string>& id) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (id.size() > free_slots_.size()) {
      return false;
    }

    for (const auto& item : id) {
      auto it = free_slots_.begin();
      id_[item] = {*it}; // Assign the slot number to this ID
      free_slots_.erase(it);
    }
    return true;
  }

  std::vector<size_t> alloc(
      const std::vector<std::string>& names,
      const std::vector<size_t>& kv_num_page) {
    size_t total = std::accumulate(kv_num_page.begin(), kv_num_page.end(), 0);

    std::vector<size_t> result;
    result.reserve(total);
    std::lock_guard<std::mutex> lock(mutex_);
    if (total < free_slots_.size()) {
      return {};
    }
    for (size_t i = 0; i < names.size(); ++i) {
      auto allocated = raw_alloc(kv_num_page[i]);
      id_[names[i]] = allocated;
      result.insert(result.end(), allocated.begin(), allocated.end());
    }
    return result;
  }

  std::vector<int> alloc(size_t need_size) {
    std::vector<int> allocated;
    allocated.reserve(need_size); // 预分配内存

    std::lock_guard<std::mutex> lock(mutex_);
    if (free_slots_.size() < need_size)
      return {};

    auto it = free_slots_.begin();
    while (allocated.size() < need_size) {
      allocated.push_back(*it);
      it = free_slots_.erase(it);
    }

    return allocated;
  }

  size_t available_size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return free_slots_.size();
  }

 private:
  size_t max_num_;
  std::unordered_set<int> free_slots_;
  mutable std::mutex mutex_;
  std::unordered_map<std::string, std::unordered_set<int>> id_;

 private:
  std::unordered_set<int> raw_alloc(int size) {
    if (size > free_slots_.size()) {
      throw std::runtime_error("not enough solts");
    }

    std::unordered_set<int> allocated;
    allocated.reserve(size);
    while (size-- > 0) {
      auto it = free_slots_.begin();
      allocated.insert(*it);
      free_slots_.erase(it);
    }
    return allocated;
  }
};

class ReqPagePool {
 public:
  ReqPagePool(size_t max_num_req, size_t max_num_page)
      : req_pool_(max_num_req), page_pool_(max_num_page) {}

  std::vector<size_t> prefill_alloc(
      const std::vector<std::string>& names,
      const std::vector<size_t>& kv_num_page) {
    if (!req_pool_.alloc(names)) {
      return {};
    }
    return page_pool_.alloc(names, kv_num_page);
  }

  void free(const std::string& req) {
    page_pool_.free(req);
    req_pool_.free(req);
  }

  void add_more_page(size_t num_added_slots) {
    page_pool_.add_more_slots(num_added_slots);
  }

  size_t available_pages() {
    return page_pool_.available_size();
  }
  size_t avaliable_ids() {
    return req_pool_.available_size();
  }

 private:
  ThreadSafeSlots req_pool_;
  ThreadSafeSlots page_pool_;
};
} // namespace torchpipe
