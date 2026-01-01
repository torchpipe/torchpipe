#include "omniback/builtin/cat_split.hpp"
#include "omniback/core/task_keys.hpp"
#include "omniback/helper/base_logging.hpp"
#include "omniback/helper/macro.h"

constexpr auto EXPECTED_DEPENDENCIES = 3;

namespace omniback {
void CatSplit::post_init(
    const std::unordered_map<std::string, std::string>& config,
    const dict& kwargs) {
  OMNI_ASSERT(
      base_dependencies_.size() == EXPECTED_DEPENDENCIES,
      "CatSplit requires exactly " + std::to_string(EXPECTED_DEPENDENCIES) +
          " comma-separated backends, but received " +
          std::to_string(base_dependencies_.size()));

  OMNI_ASSERT(
      base_dependencies_[0]->max() == std::numeric_limits<uint32_t>::max(),
      "CatSplit requires a concating backend with max() == "
      "std::numeric_limits<uint32_t>::max()");
  OMNI_ASSERT(
      base_dependencies_[2]->max() == std::numeric_limits<uint32_t>::max(),
      "CatSplit requires a spliting backend with max() == "
      "std::numeric_limits<uint32_t>::max()");
  SPDLOG_INFO("CatSplit: range=[{}, {}]", min_, max_);
}

void CatSplit::impl_forward(const std::vector<dict>& data) {
  // first stage: concatenate
  base_dependencies_[0]->forward(data);
  auto iter = data[0]->find(TASK_RESULT_KEY);
  OMNI_ASSERT(
      iter != data[0]->end(),
      "CatSplit requires a result key in the first input");
  (*data[0])[TASK_DATA_KEY] = iter->second;
  data[0]->erase(TASK_RESULT_KEY);

  // second stage: batching inference
  base_dependencies_.at(1)->forward({data.at(0)});
  iter = data[0]->find(TASK_RESULT_KEY);
  OMNI_ASSERT(
      iter != data[0]->end(),
      "CatSplit requires a result key in the second input");
  (*data[0])[TASK_DATA_KEY] = iter->second;
  data[0]->erase(TASK_RESULT_KEY);

  // third stage: Split
  base_dependencies_.at(2)->forward(data);
}

std::pair<uint32_t, uint32_t> CatSplit::update_min_max(
    const std::vector<Backend*>& depends) {
  return {depends.at(1)->min(), depends.at(1)->max()};
}

std::vector<uint32_t> CatSplit::set_init_order(uint32_t max_range) const {
  OMNI_ASSERT(max_range == 3);
  return {1, 0, 2};
}

OMNI_REGISTER_BACKEND(CatSplit);

} // namespace omniback