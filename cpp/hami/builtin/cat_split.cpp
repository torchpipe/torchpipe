#include "hami/builtin/cat_split.hpp"
#include "hami/helper/macro.h"
#include "hami/core/task_keys.hpp"
#include "hami/helper/base_logging.hpp"

constexpr auto EXPECTED_DEPENDENCIES = 3;

namespace hami {
void CatSplit::post_init(
    const std::unordered_map<std::string, std::string>& config,
    const dict& dict_config) {
    HAMI_ASSERT(base_dependencies_.size() == EXPECTED_DEPENDENCIES,
                "CatSplit requires exactly " +
                    std::to_string(EXPECTED_DEPENDENCIES) +
                    " comma-separated backends, but received " +
                    std::to_string(base_dependencies_.size()));

    HAMI_ASSERT(
        base_dependencies_[0]->max() == std::numeric_limits<size_t>::max(),
        "CatSplit requires a concating backend with max() == "
        "std::numeric_limits<size_t>::max()");
    HAMI_ASSERT(
        base_dependencies_[2]->max() == std::numeric_limits<size_t>::max(),
        "CatSplit requires a spliting backend with max() == "
        "std::numeric_limits<size_t>::max()");
    SPDLOG_INFO("CatSplit: range=[{}, {}]", min_, max_);
}

void CatSplit::forward(const std::vector<dict>& data) {
    // first stage: concatenate
    base_dependencies_[0]->forward(data);
    auto iter = data[0]->find(TASK_RESULT_KEY);
    HAMI_ASSERT(iter != data[0]->end(),
                "CatSplit requires a result key in the first input");
    (*data[0])[TASK_DATA_KEY] = iter->second;
    data[0]->erase(TASK_RESULT_KEY);

    // second stage: batching inference
    base_dependencies_.at(1)->forward({data.at(0)});
    iter = data[0]->find(TASK_RESULT_KEY);
    HAMI_ASSERT(iter != data[0]->end(),
                "CatSplit requires a result key in the second input");
    (*data[0])[TASK_DATA_KEY] = iter->second;
    data[0]->erase(TASK_RESULT_KEY);

    // third stage: Split
    base_dependencies_.at(2)->forward(data);
}

std::pair<size_t, size_t> CatSplit::update_min_max(
    const std::vector<Backend*>& depends) {
    return {depends.at(1)->min(), depends.at(1)->max()};
}

std::vector<size_t> CatSplit::set_init_order(size_t max_range) const {
    HAMI_ASSERT(max_range == 3);
    return {1, 0, 2};
}

HAMI_REGISTER_BACKEND(CatSplit);

}  // namespace hami