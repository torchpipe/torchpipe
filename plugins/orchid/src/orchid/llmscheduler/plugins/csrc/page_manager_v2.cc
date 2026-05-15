#include <tvm/ffi/c_api.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/extra/stl.h> // Enable STL conversion
#include <vector>
#include <mutex>
#include <map>
#include <stdexcept>
#include <cstring>
#include <iostream>
#include <memory>

// Minimal implementation of PageManager using std::vector and map
// This is a CPU-only implementation for demonstration.
// It manages logical -> physical page mapping.

namespace custom {

using namespace tvm::ffi;

// Simple Page Manager
class PageManager {
public:
    PageManager(int max_pages) : max_pages_(max_pages) {
        for (int i = 0; i < max_pages; ++i) {
            free_pages_.push_back(i);
        }
    }

    std::vector<int32_t> GetPages(int req_id, int num_needed) {
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

    void Free(int req_id) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (req_pages_.count(req_id)) {
            auto& pages = req_pages_[req_id];
            free_pages_.insert(free_pages_.end(), pages.begin(), pages.end());
            req_pages_.erase(req_id);
        }
    }

    void Reset() {
        std::lock_guard<std::mutex> lock(mutex_);
        req_pages_.clear();
        free_pages_.clear();
        for (int i = 0; i < max_pages_; ++i) free_pages_.push_back(i);
    }

private:
    int max_pages_;
    std::vector<int32_t> free_pages_;
    std::map<int, std::vector<int32_t>> req_pages_;
    std::mutex mutex_;
};

static std::unique_ptr<PageManager> PM;

void Init(int max_pages) {
    PM = std::make_unique<PageManager>(max_pages);
}

// Return std::vector<int32_t> directly. 
// tvm/ffi/extra/stl.h enables automatic conversion to Array
std::vector<int32_t> GetPagesFunc(int req_id, int num_needed) {
    if (!PM) Init(4096); // Default fallback
    return PM->GetPages(req_id, num_needed);
}

void FreeFunc(int req_id) {
    if (PM) PM->Free(req_id);
}

void ResetFunc() {
    if (PM) PM->Reset();
}

// Register global functions using reflection::GlobalDef
TVM_FFI_STATIC_INIT_BLOCK() {
    namespace refl = tvm::ffi::reflection;
    refl::GlobalDef()
        .def("custom.init", Init)
        .def("custom.get_pages", GetPagesFunc)
        .def("custom.free", FreeFunc)
        .def("custom.reset", ResetFunc);
}

} // namespace custom
