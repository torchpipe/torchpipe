
#include "hami/builtin/basic_backends.hpp"
#include "hami/core/reflect.h"
#include "hami/helper/macro.h"
#include "hami/helper/base_logging.hpp"
#include "hami/helper/string.hpp"
#include "hami/core/helper.hpp"
#include "hami/builtin/select.hpp"

namespace hami {

void Select::post_init(
    const std::unordered_map<std::string, std::string>& config,
    const dict& kwargs) {
    select_ = select_impl();
    HAMI_ASSERT(base_dependencies_.size() >= 2);
}

void Select::impl_forward(const std::vector<dict>& input_output) {
    DictHelper dicts_guard(input_output);
    dicts_guard.keep_alive(TASK_DATA_KEY).erase(TASK_RESULT_KEY);
    std::unordered_map<size_t, std::vector<dict>> inputs;

    for (const auto& input : input_output) {
        size_t key = select_(input);
        HAMI_ASSERT(key < base_dependencies_.size());
        if (inputs.find(key) == inputs.end()) {
            inputs.insert({{key, {input}}});
        } else {
            inputs[key].push_back(input);
        }
    }
    for (const auto& [key, data] : inputs) {
        base_dependencies_[key]->safe_forward(data);
    }
}

std::pair<size_t, size_t> Select::update_min_max(
    const std::vector<Backend*>& depends) {
    // union
    size_t max_value = 1;
    size_t min_value = std::numeric_limits<size_t>::max();

    for (Backend* depend : depends) {
        min_value = std::min(min_value, depend->min());
        max_value = std::max(max_value, depend->max());
    }

    HAMI_ASSERT(min_value <= max_value);

    // // check bouble
    // if (min_value == 1)
    //   return {min_value, max_value};
    return {min_value, max_value};
}

// HAMI_REGISTER(Backend, Select);

}  // namespace hami
