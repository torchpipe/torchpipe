
#include <condition_variable>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "omniback/helper/base_logging.hpp"
#include "omniback/schedule/schedule_states.hpp"

// InstanceDispatcher, Batching (动态dependency)
// forward instance="node_name.0"
namespace omniback {
// IoCV0[SharedInstancesState,InstanceDispatcher,Batching;DI_v0[Batching,
// InstanceDispatcher]]

bool InstancesState::wait_for(size_t req_size, size_t timeout) {
  std::unique_lock<std::mutex> lock(mtx_);
  std::optional<size_t> best_match;
  cv_.wait_for(
      lock,
      std::chrono::milliseconds(timeout),
      [this, req_size, &best_match]() {
        size_t min_size = std::numeric_limits<uint32_t>::max();

        for (const auto& ins : available_instances_) {
          const auto& [lower, upper] = instances_.at(ins);
          if (lower <= req_size && req_size <= upper) {
            if (upper <= min_size) {
              min_size = upper;
              best_match = ins;
            }
          }
        }
        return bool(best_match);
      });
  return bool(best_match);
}

std::optional<uint32_t> InstancesState::query_available(
    uint32_t req_size,
    size_t timeout,
    bool lock_queried,
    std::string node_name) {
  std::unique_lock<std::mutex> lock(mtx_);
  std::optional<uint32_t> best_match;
  cv_.wait_for(
      lock,
      std::chrono::milliseconds(timeout),
      [this, lock_queried, req_size, &best_match]() {
        uint32_t min_size = std::numeric_limits<uint32_t>::max();

        for (const auto& ins : available_instances_) {
          const auto& [lower, upper] = instances_.at(ins);
          if (lower <= req_size && req_size <= upper) {
            if (upper <= min_size) {
              min_size = upper;
              best_match = ins;
            }
          }
        }

        if (best_match && lock_queried) {
          available_instances_.erase(*best_match);
          locked_available_instances_.insert(*best_match);
        }
        return bool(best_match);
      });
  if (!best_match) {
    auto min_v = available_instances_.empty()
        ? 0
        : instances_.at(*available_instances_.begin()).first;
    auto max_v = available_instances_.empty()
        ? 0
        : instances_.at(*available_instances_.begin()).second;
    SPDLOG_DEBUG(
        "query_available timeout. req_size: {}, available_instances_: {} min={} max={} node_name={}",
        req_size,
        available_instances_.size(),
        min_v,
        max_v,
        node_name);
  }
  return best_match;
}

void InstancesState::add_and_set_range(
    size_t handle,
    size_t min_value,
    size_t max_value) {
  std::unique_lock<std::mutex> lock(mtx_);
  instances_[handle] = {min_value, max_value};
  available_instances_.insert(handle);
}

} // namespace omniback