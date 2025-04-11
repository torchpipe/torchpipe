
#include <memory>
#include <string>
#include <vector>
#include <mutex>
#include <condition_variable>
#include <unordered_set>
#include <unordered_map>
#include <optional>

#include "hami/schedule/schedule_states.hpp"
#include "hami/helper/base_logging.hpp"

// InstanceDispatcher, Batching (动态dependency)
// forward instance="node_name.0"
namespace hami {
// IoCV0[SharedInstancesState,InstanceDispatcher,Batching;DI_v0[Batching,
// InstanceDispatcher]]
std::optional<size_t> InstancesState::query_avaliable(
    size_t req_size,
    size_t timeout,
    bool lock_queried) {
  std::unique_lock<std::mutex> lock(mtx_);
  std::optional<size_t> best_match;
  cv_.wait_for(
      lock,
      std::chrono::milliseconds(timeout),
      [this, lock_queried, req_size, &best_match]() {
        size_t min_size = std::numeric_limits<size_t>::max();

        for (const auto& ins : avaliable_instances_) {
          const auto& [lower, upper] = instances_.at(ins);
          if (lower <= req_size && req_size <= upper) {
            if (upper <= min_size) {
              min_size = upper;
              best_match = ins;
            }
          }
        }

        if (best_match && lock_queried) {
          avaliable_instances_.erase(*best_match);
          locked_avaliable_instances_.insert(*best_match);
        }
        return bool(best_match);
      });
  if (!best_match) {
    auto min_v = avaliable_instances_.empty()
        ? 0
        : instances_.at(*avaliable_instances_.begin()).first;
    auto max_v = avaliable_instances_.empty()
        ? 0
        : instances_.at(*avaliable_instances_.begin()).second;
    SPDLOG_WARN(
        "query_avaliable timeout. req_size: {}, avaliable_instances_: {} min={} max={}",
        req_size,
        avaliable_instances_.size(),
        min_v,
        max_v);
  }
  return best_match;
}

void InstancesState::add_and_set_range(
    size_t handle,
    size_t min_value,
    size_t max_value) {
  std::unique_lock<std::mutex> lock(mtx_);
  instances_[handle] = {min_value, max_value};
  avaliable_instances_.insert(handle);
}

} // namespace hami