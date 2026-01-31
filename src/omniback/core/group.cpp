#include <unordered_map>
#include <vector>
#include <mutex>
#include <utility>
#include <functional>
#include <string>

#include "omniback/core/group.hpp"
#include "omniback/helper/base_logging.hpp"

namespace om {

namespace {
// 线程安全的全局注册表
struct Registry {
    std::mutex mutex;
    std::unordered_map<
        std::string,
        std::vector<std::pair<std::string, std::function<bool()>>>
    > map;
};

Registry& get_registry() {
    static Registry registry;
    return registry;
}
} // namespace

void clear_groups() {
    auto& reg = get_registry();
    std::lock_guard<std::mutex> lock(reg.mutex);
    reg.map.clear();
}

bool register_backend_group(
    const std::string& backend_name,
    const std::string& grp_name,
    std::function<bool()> callback)
{
    auto& reg = get_registry();
    std::lock_guard<std::mutex> lock(reg.mutex);
    
    auto& callbacks = reg.map[backend_name]; // 自动创建空vector（若不存在）
    // const bool is_first_registration = callbacks.empty();
    
    callbacks.emplace_back(grp_name, std::move(callback));
    
    return true;
}

bool try_callback_from_group(const std::string& backend_name) {
    auto& reg = get_registry();
    std::lock_guard<std::mutex> lock(reg.mutex);
    
    auto it = reg.map.find(backend_name);
    if (it == reg.map.end()) {
        SPDLOG_INFO("Has no roles for {}", backend_name);
        return false;
    }
    
    auto& callbacks = it->second;
    size_t total = callbacks.size();


    size_t attempt = 1;
    for (auto rit = callbacks.rbegin(); rit != callbacks.rend(); ++rit, ++attempt) {
        const auto& [grp_name, callback] = *rit;
        SPDLOG_INFO("jit({}/{}): (group: {}, backend: {})", attempt, total, grp_name, backend_name);

        bool result = callback();

        if (result) {
            return true; // 任一成功即返回 true（短路）
        }
        SPDLOG_INFO("jit: failed.\n");
    }
    
    return false;
}

} // namespace om