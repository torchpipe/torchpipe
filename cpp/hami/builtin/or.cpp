
#include "hami/builtin/or.hpp"

#include "hami/builtin/basic_backends.hpp"
#include "hami/core/helper.hpp"
#include "hami/core/reflect.h"
#include "hami/core/task_keys.hpp"
#include "hami/helper/base_logging.hpp"
#include "hami/helper/macro.h"
#include "hami/helper/string.hpp"
#include "hami/schedule/aspect.hpp"

namespace hami {

void Or::post_init(
    const std::unordered_map<std::string, std::string>&,
    const dict&) {
  HAMI_ASSERT(base_dependencies_.size() >= 2);
}
void Or::impl_forward(const std::vector<dict>& input_output) {
  DictHelper dicts_guard(input_output);
  dicts_guard.keep(TASK_DATA_KEY).erase(TASK_RESULT_KEY);

  std::set<std::size_t> break_index;
  for (std::size_t i = 0; i < base_dependencies_.size(); ++i) {
    // filters
    std::vector<dict> valid_inputs;
    if (i == 0)
      valid_inputs = input_output;
    else if (i == 1) {
      for (std::size_t j = 0; j < input_output.size(); ++j) {
        const auto input_dict = input_output[j];
        auto iter = input_dict->find(TASK_RESULT_KEY);
        if (iter != input_dict->end()) {
          (*input_dict)[TASK_DATA_KEY] = iter->second;
          input_dict->erase(TASK_RESULT_KEY);
        }
        valid_inputs.push_back(input_dict);
      }
    } else
      for (std::size_t j = 0;
           j < input_output.size() && break_index.count(j) == 0;
           ++j) {
        const auto input_dict = input_output[j];
        auto iter = input_dict->find(TASK_RESULT_KEY);
        if (iter != input_dict->end()) {
          (*input_dict)[TASK_DATA_KEY] = iter->second;
          input_dict->erase(TASK_RESULT_KEY);
          valid_inputs.push_back(input_dict);
        } else {
          break_index.insert(j);
          if (break_index.size() == input_output.size())
            return;
        }
      }
    if (valid_inputs.empty()) {
      SPDLOG_INFO("Or: valid_inputs.empty()");
      return;
    }

    base_dependencies_[i]->safe_forward(valid_inputs);
  }
}
HAMI_REGISTER(Backend, Or, "Or, S");

} // namespace hami
