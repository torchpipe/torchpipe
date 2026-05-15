#pragma once
#include <vector>
#include <map>
#include <mutex>
#include <memory>
#include <stdexcept>

namespace custom {

class PageManager {
public:
    explicit PageManager(int max_pages);
    std::vector<int32_t> GetPages(int req_id, int num_needed);
    void Free(int req_id);
    void Reset();
    int GetMaxPages() const { return max_pages_; }

private:
    int max_pages_;
    std::vector<int32_t> free_pages_;
    std::map<int, std::vector<int32_t>> req_pages_;
    std::mutex mutex_;
};

} // namespace custom
