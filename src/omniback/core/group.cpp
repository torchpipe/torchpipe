#include <unordered_map>

#include "omniback/core/group.hpp"


namespace om{

namespace {
auto& get_registry() {
  static std::unordered_map<std::string,
          std::pair<std::string, std::function<void()>>>
      registry;
  return registry;
}
}

void clear_groups(){
    auto &reg = get_registry();
    reg.clear();
}

bool register_backend_group(const std::string &backend_name, 
    const std::string &grp_name, 
    std::function<void()> callback){
        auto &reg = get_registry();
        auto [it, inserted] = reg.insert_or_assign(
            backend_name, 
            std::make_pair(grp_name, std::move(callback))
        );
    return inserted;
}

bool try_callback_from_group(const std::string &backend_name){
    auto &reg = get_registry();
    auto it = reg.find(backend_name);
    if (it == reg.end()) {
        return false;
    }
    auto [grp_name, callback] = it->second;
    callback();
    return true;
}


}