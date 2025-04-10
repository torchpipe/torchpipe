#include "hami/builtin/condition.hpp"

#include "hami/helper/base_logging.hpp"
#include "hami/helper/string.hpp"
#include "hami/core/helper.hpp"
#include "hami/core/parser.hpp"

namespace hami {

void Condition::pre_init(
    const std::unordered_map<std::string, std::string>& config,
    const dict& kwargs) {
  condition_ = condition_impl();
  init_dep_impl(config, kwargs);
}

class HasKeyV0 : public Condition {
 public:
  void init_dep_impl(
      const std::unordered_map<std::string, std::string>& config,
      const dict& kwargs) override {
    if (config.find("key") != config.end()) {
      key_ = config.at("key");
    }
  }

  virtual std::function<bool(const dict&)> condition_impl() const override {
    return
        [this](const dict& input) { return input->find(key_) != input->end(); };
  }

 protected:
  std::string key_{"key"};
};

class NotHasKeyV0 : public HasKeyV0 {
  virtual std::function<bool(const dict&)> condition_impl() const override {
    return
        [this](const dict& input) { return input->find(key_) == input->end(); };
  }
};

HAMI_REGISTER(Backend, HasKeyV0);
HAMI_REGISTER(Backend, NotHasKeyV0);

void NotHasKey::impl_init(
    const std::unordered_map<std::string, std::string>& params,
    const dict& options) {
  auto [args, kwargs] = parser_v2::get_args_kwargs(this, "NotHasKey", params);
  HAMI_ASSERT(args.size() >= 1);
  key_ = args[0];
  auto dep = get_dependency_name_force(this, kwargs);
  dependency_ = init_backend(dep, kwargs, options);
  HAMI_ASSERT(dependency_, "Dependency not found: " + dep);
}
void NotHasKey::forward(const dict& io) {
  if (io->find(key_) == io->end()) {
    dependency_->forward({io});
  } else {
    SPDLOG_DEBUG("NotHasKey: {} exists, skip", key_);
    io->insert({TASK_RESULT_KEY, io->at(TASK_DATA_KEY)});
  }
}
HAMI_REGISTER(Backend, NotHasKey);

void HasKey::impl_init(
    const std::unordered_map<std::string, std::string>& params,
    const dict& options) {
  auto [args, kwargs] = parser_v2::get_args_kwargs(this, "HasKey", params);
  HAMI_ASSERT(args.size() >= 1);
  key_ = args[0];
  auto dep = get_dependency_name_force(this, kwargs);
  // auto deps = str::str_split(dep, ',');
  parser_v2::Parser parser;
  auto deps = parser.split_by_delimiter(dep, ',');

  HAMI_ASSERT(deps.size() >= 1, "error dep: " + dep);

  dependency_a_ = init_backend(deps[0], kwargs, options);
  HAMI_ASSERT(dependency_a_, "Dependency not found: " + dep);

  if (deps.size() >= 2) {
    dependency_b_ = init_backend(deps[1], kwargs, options);
    HAMI_ASSERT(dependency_b_, "Dependency not found: " + dep);
  }
}
void HasKey::forward(const dict& io) {
  if (io->find(key_) != io->end()) {
    dependency_a_->forward({io});
  } else if (dependency_b_) {
    dependency_b_->forward({io});
  } else {
    SPDLOG_DEBUG("HasKey: {} exists, skip", key_);
    io->insert({TASK_RESULT_KEY, io->at(TASK_DATA_KEY)});
  }
}
HAMI_REGISTER(Backend, HasKey);

} // namespace hami