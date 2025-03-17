
#include <memory>
#include <string>
#include <vector>
#include <mutex>
#include <condition_variable>
#include <unordered_set>
#include <unordered_map>
#include <optional>

#include "hami/schedule/schedule_states.hpp"

// InstanceDispatcher, Batching (动态dependency)
// forward instance="node_name.0"
namespace hami
{
    // IoCV0[SharedInstancesState,InstanceDispatcher,Batching;DI_v0[Batching,
    // InstanceDispatcher]]

    void InstancesState::add_and_set_range(size_t handle, size_t min_value,
                                           size_t max_value)
    {
        std::unique_lock<std::mutex> lock(mtx_);
        instances_[handle] = {min_value, max_value};
        avaliable_instances_.insert(handle);
    }

} // namespace hami