#include "omniback/builtin/result_parser.hpp"
#include "omniback/core/reflect.h"
#include "omniback/core/task_keys.hpp"
#include "omniback/helper/base_logging.hpp"
#include "omniback/helper/string.hpp"

namespace omniback {

void ResultParser::pre_init(
    const std::unordered_map<std::string, std::string>& config,
    const dict& kwargs) {
  parser_ = parser_impl();
  init_dep_impl(config, kwargs);
}

class ThrowIfNoResult : public ResultParser {
 public:
  virtual std::function<void(const dict& data)> parser_impl() const override {
    return [](const dict& input) {
      if (input->find(TASK_RESULT_KEY) == input->end()) {
        throw std::runtime_error("ThrowIfNoResult: No result found");
      }
    };
  }
};

OMNI_REGISTER(Backend, ThrowIfNoResult);

class RuntimeError : public BackendOne {
 public:
  void forward(const dict& io) override {
    throw std::runtime_error("in RuntimeError backend");
  }
};

OMNI_REGISTER(Backend, RuntimeError);

} // namespace omniback