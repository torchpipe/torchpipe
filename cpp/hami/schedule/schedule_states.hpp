#pragma once

#include <memory>
#include <string>
#include <vector>
#include <mutex>
#include <condition_variable>
#include <unordered_set>
#include <unordered_map>
#include <optional>

// InstanceDispatcher, Batching (动态dependency)
// forward instance="node_name.0"
namespace hami {
class InstancesState {
 public:
  void add(size_t index, size_t min_value, size_t max_value) {
    std::unique_lock<std::mutex> lock(mtx_);
    instances_[index] = {min_value, max_value};
    empty_instances_.insert(index);
  }

  bool wait_resource(size_t req_size, size_t& index, size_t timeout) {
    std::unique_lock<std::mutex> lock(mtx_);

    return cv_.wait_for(lock, std::chrono::milliseconds(timeout), [this, req_size, &index] {
      std::optional<size_t> best_match;
      size_t min_size = std::numeric_limits<size_t>::max();

      for (const auto& ins : empty_instances_) {
        const auto& [lower, upper] = instances_[ins];
        if (lower <= req_size && req_size <= upper) {
          if (upper <= min_size) {
            min_size = upper;
            best_match = ins;
          }
        }
      }

      if (best_match) {
        index = *best_match;
        empty_instances_.erase(index);
        return true;
      }
      return false;
    });
  }

  void finished_instance(size_t index) {
    {
      std::lock_guard<std::mutex> lock(mtx_);
      empty_instances_.insert(index);
    }
    cv_.notify_all();
  }

 private:
  mutable std::mutex mtx_;

  std::condition_variable cv_;

  std::unordered_set<size_t> empty_instances_;
  std::unordered_map<size_t, std::pair<size_t, size_t>> instances_;
};

}  // namespace hami