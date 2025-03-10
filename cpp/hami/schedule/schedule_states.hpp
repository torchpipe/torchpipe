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
// IoC[SharedInstancesState,InstanceDispatcher,Batching;DI[Batching,
// InstanceDispatcher]]
constexpr auto TASK_RESOURCE_STATE_KEY = "resource_state";
// class ResourceState {
//    public:
//     // @return handle
//     virtual std::optional<size_t> wait_avaliable(size_t req_size,
//                                                  size_t timeout,
//                                                  bool erase_avaliable) = 0;
// };

class InstancesState {
   public:
    void add_and_set_range(size_t handle, size_t min_value, size_t max_value) {
        std::unique_lock<std::mutex> lock(mtx_);
        instances_[handle] = {min_value, max_value};
        avaliable_instances_.insert(handle);
    }

    void add(size_t handle) {
        {
            std::unique_lock<std::mutex> lock(mtx_);
            avaliable_instances_.insert(handle);
        }

        cv_.notify_all();
    }

    std::optional<size_t> query_avaliable(size_t req_size, size_t timeout,
                                          bool lock_queried) {
        std::unique_lock<std::mutex> lock(mtx_);
        std::optional<size_t> best_match;
        cv_.wait_for(lock, std::chrono::milliseconds(timeout),
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
        return best_match;
    }

    void remove_lock(size_t handle) {
        {
            std::lock_guard<std::mutex> lock(mtx_);
            locked_avaliable_instances_.erase(handle);
            avaliable_instances_.insert(handle);
        }
        cv_.notify_all();
    }

    void notify_all() { cv_.notify_all(); }

   private:
    mutable std::mutex mtx_;

    std::condition_variable cv_;

    std::unordered_set<size_t> avaliable_instances_;
    std::unordered_set<size_t> locked_avaliable_instances_;
    std::unordered_map<size_t, std::pair<size_t, size_t>> instances_;
};

}  // namespace hami