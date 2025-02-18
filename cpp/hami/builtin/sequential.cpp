
#include <unordered_set>
#include "hami/builtin/basic_backends.hpp"
#include "hami/core/reflect.h"
#include "hami/helper/macro.h"
#include "hami/helper/base_logging.hpp"
#include "hami/helper/string.hpp"
#include "hami/builtin/sequential.hpp"
#include "hami/core/helper.hpp"
#include "hami/core/task_keys.hpp"

namespace hami {

void Sequential::forward(const std::vector<dict>& input_output) {
  DictHelper dicts_guard(input_output);
  dicts_guard.keep(TASK_DATA_KEY)
      .erase(TASK_RESULT_KEY);  // to keey the storage of TASK_DATA_KEY. This tensor is
  // may be created in another stream
  std::unordered_set<std::size_t> break_index;
  for (std::size_t i = 0; i < base_dependencies_.size(); ++i) {
    // filters
    std::vector<dict> valid_inputs;
    if (i == 0)
      valid_inputs = input_output;
    else {
      bool or_filter = base_config_.at(i).find("or") != base_config_.at(i).end() &&
                       (base_config_.at(i)["or"] == "1");
      for (std::size_t j = 0; j < input_output.size() && break_index.count(j) == 0; ++j) {
        const auto input_dict = input_output[j];
        auto iter = input_dict->find(TASK_RESULT_KEY);
        if (iter != input_dict->end()) {
          (*input_dict)[TASK_DATA_KEY] = iter->second;
          input_dict->erase(iter);
          valid_inputs.push_back(input_dict);
        } else if (or_filter) {
          valid_inputs.push_back(input_dict);
        } else {
          dicts_guard.erase(TASK_RESULT_KEY);
          throw std::runtime_error("Sequential: no result in " + base_config_[i].at("backend"));
          break_index.insert(j);
          if (break_index.size() == input_output.size()) return;
        }
      }
    }

    if (valid_inputs.empty()) {
      SPDLOG_INFO("Sequential: valid_inputs.empty()");
      return;
    }

    base_dependencies_[i]->safe_forward(valid_inputs);
  }
}
HAMI_REGISTER(Backend, Sequential, "Sequential, S");

}  // namespace hami
