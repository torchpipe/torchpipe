#include "page_manager.h"
#include <stdexcept>
#include <algorithm>

namespace custom {

PageManager::PageManager(int max_pages) : max_pages_(max_pages) {
    free_pages_.reserve(max_pages);
    for (int i = 0; i < max_pages; ++i) {
        free_pages_.push_back(i);
    }
}

std::vector<int32_t> PageManager::GetPages(int req_id, int num_needed) {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<int32_t>& current = req_pages_[req_id];
    
    if (current.size() >= num_needed) {
        return {current.begin(), current.begin() + num_needed};
    }
    
    int needed = num_needed - current.size();
    if (free_pages_.size() < needed) {
        throw std::runtime_error("OOM: Not enough pages");
    }
    
    for (int i = 0; i < needed; ++i) {
        current.push_back(free_pages_.back());
        free_pages_.pop_back();
    }
    return current;
}

void PageManager::Free(int req_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (req_pages_.count(req_id)) {
        auto& pages = req_pages_[req_id];
        free_pages_.insert(free_pages_.end(), pages.begin(), pages.end());
        req_pages_.erase(req_id);
    }
}

void PageManager::Reset() {
    std::lock_guard<std::mutex> lock(mutex_);
    req_pages_.clear();
    free_pages_.clear();
    free_pages_.reserve(max_pages_);
    for (int i = 0; i < max_pages_; ++i) free_pages_.push_back(i);
}

} // namespace custom
