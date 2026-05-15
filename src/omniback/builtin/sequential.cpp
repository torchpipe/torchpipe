
#include "omniback/builtin/sequential.hpp"

#include <unordered_set>

#include "omniback/builtin/basic_backends.hpp"
#include "omniback/core/helper.hpp"
#include "omniback/core/parser.hpp"
#include "omniback/core/reflect.h"
#include "omniback/core/task_keys.hpp"
#include "omniback/helper/base_logging.hpp"
#include "omniback/helper/macro.h"
#include "omniback/helper/string.hpp"

namespace om {

void SequentialV0::impl_forward(const std::vector<dict>& io) {
  DictHelper dicts_guard(io);
  dicts_guard.keep(TASK_DATA_KEY)
      .erase(TASK_RESULT_KEY); // to keey the storage of TASK_DATA_KEY. This
                               // tensor is
  // may be created in another stream
  std::unordered_set<std::size_t> break_index;
  for (std::size_t i = 0; i < base_dependencies_.size(); ++i) {
    // filters
    std::vector<dict> valid_inputs;
    const auto& curr_config = base_config_.at(i);
    if (i == 0)
      valid_inputs = io;
    else {
      const bool or_filter =
          curr_config.find("or") != curr_config.end() &&
          (curr_config.at("or") == "1");
      for (std::size_t j = 0; j < io.size() && break_index.count(j) == 0; ++j) {
        const auto& input_dict = io[j];
        auto iter = input_dict->find(TASK_RESULT_KEY);
        if (iter != input_dict->end()) {
          (*input_dict)[TASK_DATA_KEY] = iter->second;
          input_dict->erase(iter);
          valid_inputs.push_back(input_dict);
        } else if (or_filter) {
          valid_inputs.push_back(input_dict);
        } else {
          dicts_guard.erase(TASK_RESULT_KEY);
          throw std::runtime_error(
              "SequentialV0: no result in " +
              base_config_[i - 1].at("backend"));
          break_index.insert(j);
          if (break_index.size() == io.size())
            return;
        }
      }
    }

    if (valid_inputs.empty()) {
      SPDLOG_INFO("SequentialV0: valid_inputs.empty()");
      return;
    }

    base_dependencies_[i]->safe_forward(valid_inputs);
  }
}
OMNI_REGISTER(Backend, SequentialV0, "SequentialV0, S_v0");

void Sequential::impl_custom_init(
    const std::unordered_map<std::string, std::string>& params,
    const dict& options) {
  parser_v2::Parser parser;

  backends_.resize(backend_cfgs_.size());
  filter_or_.resize(backend_cfgs_.size(), false);

  for (size_t i = 0; i < backend_cfgs_.size(); ++i) {
    const size_t ri = backend_cfgs_.size() - i - 1;

    std::unique_ptr<Backend> backend =
        init_backend(backend_cfgs_[ri], params, options);
    backends_[ri] = std::move(backend);

    // prefix
    const auto& prefix = prefix_args_kwargs_[ri];
    if (prefix.first.size() >= 1 && prefix.first[0] == "or") {
      SPDLOG_INFO("Sequential: or filter in {}", backend_cfgs_[ri]);
      filter_or_[ri] = true;
    }
  }
}

void Sequential::update_min_max() {
  // union of all backends
  max_ = std::numeric_limits<uint32_t>::max();
  min_ = 1;
  size_t num_one = 0;
  for (const auto& depend : backends_) {
    if (depend->max() == 1) {
      num_one++;
    } else {
      min_ = std::max(min_, depend->min());
      max_ = std::min(max_, depend->max());
    }
  }

  if (num_one == backends_.size()) {
    max_ = 1;
  } else if (max_ == std::numeric_limits<uint32_t>::max() && num_one != 0) {
    max_ = 1;
  }
  // else if (num_one != 0 && max_ == std::numeric_limits<uint32_t>::max()) {
  //   max_ = 1;
  // }
  // if (num_one != 0 && max_ == std::numeric_limits<uint32_t>::max()) {
  //   max_ = 1;
  // }

  SPDLOG_INFO("Sequential: min={}, max={}", min_, max_);
}

void Sequential::impl_forward(const std::vector<dict>& io) {
  DictHelper dicts_guard(io);
  // dicts_guard.keep_alive(TASK_DATA_KEY)
  //     .erase(TASK_RESULT_KEY);
  dicts_guard.erase(TASK_RESULT_KEY);
  // may be created in another stream
  std::unordered_set<std::size_t> break_index;
  for (std::size_t i = 0; i < backends_.size(); ++i) {
    // filters
    std::vector<dict> valid_inputs;
    const bool is_filter_or = filter_or_[i];
    if (i == 0)
      valid_inputs = io;
    else {
      for (std::size_t j = 0; j < io.size() && break_index.count(j) == 0; ++j) {
        const auto& input_dict = io[j];
        auto iter = input_dict->find(TASK_RESULT_KEY);
        if (iter != input_dict->end()) {
          if (!is_filter_or) {
            (*input_dict)[TASK_DATA_KEY] = iter->second;
            input_dict->erase(iter);
            valid_inputs.push_back(input_dict);
          } else {
            continue;
          }
        } else if (is_filter_or) {
          valid_inputs.push_back(input_dict);
        } else {
          // dicts_guard.erase(TASK_RESULT_KEY);
          throw std::runtime_error(
              "Sequential: no result in " + backend_cfgs_.at(i - 1));
          break_index.insert(j);
          if (break_index.size() == io.size())
            return;
        }
      }
    }

    if (valid_inputs.empty()) {
      // SPDLOG_INFO("Sequential: valid_inputs.empty()");
      continue;
    }

    // todo: add a check for the backend
    backends_[i]->safe_forward(valid_inputs);
  }
}
OMNI_REGISTER(Backend, Sequential, "Sequential, S");

} // namespace om
