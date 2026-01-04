#include <random>

#include "omniback/builtin/condition.hpp"

#include "omniback/core/helper.hpp"
#include "omniback/core/parser.hpp"
#include "omniback/helper/base_logging.hpp"
#include "omniback/helper/string.hpp"

namespace omniback {
namespace {
float generate_random_number() {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dis(0.0, 1.0);
  return dis(gen);
}
} // namespace

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

OMNI_REGISTER(Backend, HasKeyV0);
OMNI_REGISTER(Backend, NotHasKeyV0);

void NotHasKey::impl_init(
    const std::unordered_map<std::string, std::string>& params,
    const dict& options) {
  auto [args, kwargs] = parser_v2::get_args_kwargs(this, "NotHasKey", params);
  OMNI_ASSERT(args.size() >= 1);
  key_ = args[0];
  auto dep = get_dependency_name_force(this, kwargs);
  dependency_ = init_backend(dep, kwargs, options);
  OMNI_ASSERT(dependency_, "Dependency not found: " + dep);
}
void NotHasKey::forward(const dict& io) {
  if (io->find(key_) == io->end()) {
    dependency_->forward({io});
  } else {
    SPDLOG_DEBUG("NotHasKey: {} exists, skip", key_);
    io->insert({TASK_RESULT_KEY, io->at(TASK_DATA_KEY)});
  }
}
OMNI_REGISTER(Backend, NotHasKey);

void HasKey::impl_init(
    const std::unordered_map<std::string, std::string>& params,
    const dict& options) {
  auto [args, kwargs] = parser_v2::get_args_kwargs(this, "HasKey", params);
  OMNI_ASSERT(args.size() >= 1);
  key_ = args[0];
  auto dep = get_dependency_name_force(this, kwargs);
  // auto deps = str::str_split(dep, ',');
  parser_v2::Parser parser;
  auto deps = parser.split_by_delimiter(dep, ',');

  OMNI_ASSERT(deps.size() >= 1, "error dep: " + dep);

  dependency_a_ = init_backend(deps[0], kwargs, options);
  OMNI_ASSERT(dependency_a_, "Dependency not found: " + dep);

  if (deps.size() >= 2) {
    dependency_b_ = init_backend(deps[1], kwargs, options);
    OMNI_ASSERT(dependency_b_, "Dependency not found: " + dep);
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
OMNI_REGISTER(Backend, HasKey);

class Random : public Backend {
 private:
  void impl_init(
      const std::unordered_map<std::string, std::string>& params,
      const dict& options) override {
    auto [args, kwargs] = parser_v2::get_args_kwargs(this, "Random", params);
    OMNI_ASSERT(args.size() == 1);
    probability_ = std::stof(args[0]);
    auto dep = get_dependency_name_force(this, kwargs);
    // auto deps = str::str_split(dep, ',');
    parser_v2::Parser parser;
    auto deps = parser.split_by_delimiter(dep, ',');

    OMNI_ASSERT(deps.size() == 2, "error dep: " + dep);

    dependency_a_ = OMNI_INSTANCE_GET(Backend, deps[0]);
    dependency_b_ = OMNI_INSTANCE_GET(Backend, deps[1]);
    OMNI_ASSERT(dependency_a_ && dependency_b_, "Dependency not found: " + dep);
    SPDLOG_INFO(
        "Random: probability={}, dependency_a={}, dependency_b={}",
        probability_,
        deps[0],
        deps[1]);
    OMNI_ASSERT(
        probability_ >= 0.0f && probability_ <= 1.0f,
        "Random probability must be in [0, 1], got: " +
            std::to_string(probability_));
    OMNI_ASSERT(dependency_a_->max() == dependency_b_->max());
    OMNI_ASSERT(
        dependency_a_->min() == dependency_b_->min() &&
        dependency_a_->min() == 1);
  }
  [[nodiscard]] uint32_t impl_max() const override {
    return dependency_a_->max();
  }
  [[nodiscard]] uint32_t impl_min() const override {
    return dependency_a_->min();
  }

  void impl_forward(const std::vector<dict>& ios) override {
    if (generate_random_number() < probability_) {
      SPDLOG_DEBUG("Random: condition met, executing dependency_a");
      dependency_a_->forward(ios);
    } else {
      SPDLOG_DEBUG("Random: condition not met, executing dependency_b");
      dependency_b_->forward(ios);
    }
  }

 private:
  float probability_{0.5f};
  Backend* dependency_a_{nullptr};
  Backend* dependency_b_{nullptr};
};
OMNI_REGISTER(Backend, Random);

} // namespace omniback