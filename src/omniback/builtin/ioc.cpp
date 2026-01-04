
#include "omniback/core/reflect.h"
#include "omniback/core/task_keys.hpp"
#include "omniback/helper/base_logging.hpp"
#include "omniback/helper/macro.h"
#include "omniback/helper/string.hpp"

#include "omniback/builtin/ioc.hpp"
#include "omniback/helper/unique_index.hpp"

#include "omniback/builtin/control_plane.hpp"
#include <tvm/ffi/error.h>

namespace omniback {

void IoCV0::impl_init(
    const std::unordered_map<std::string, std::string>& in_config,
    const dict& in_kwargs) {
  constexpr auto default_name = "IoCV0";
  auto name = OMNI_OBJECT_NAME(Backend, this);
  if (!name) {
    name = default_name;
    SPDLOG_WARN(
        "{}::init, instance not created via reflection, using default "
        "name {}. "
        "Configure dependencies via {}::dependency",
        *name,
        *name,
        *name);
  }
  auto config = in_config;
  auto iter = config.find(*name + "::dependency");
  OMNI_ASSERT(
      iter != config.end(), "Dependency configuration missing for " + *name);

  auto backend_setting = iter->second;
  // SPDLOG_INFO("IoCV0: {}", backend_setting);
  config.erase(iter);

  std::vector<std::string> phases = str::items_split(backend_setting, ';');
  OMNI_ASSERT(phases.size() == 2, "IoCV0 requires two phases separated by ';'");

  auto kwargs = in_kwargs ? in_kwargs : make_dict();
  init_phase(phases[0], config, kwargs); // Initialization phase

  // std::unordered_map<std::string, Backend*> backend_map;
  std::unordered_set<std::string> keys;
  for (size_t i = 0; i < base_config_.size(); ++i) {
    const auto& item = base_config_[i];
    auto main_backend = item.at("backend");
    TVM_FFI_ICHECK(keys.count(main_backend) == 0)
        << "Duplicate backend name detected during initialization "
           "parsing: " <<
            main_backend;

    keys.insert(main_backend);
    size_t find_start = 0;
    std::unordered_map<void*, std::string> reg_backends;
    while (phases[1].find(main_backend, find_start) != std::string::npos) {
      std::string register_name;
      if (reg_backends.find(base_dependencies_[i].get()) ==
          reg_backends.end()) {
        register_name = "ioc." + main_backend +
            "." + // std::to_string(i) + "." +
            std::to_string(get_unique_index());
        SPDLOG_DEBUG("register {} {} {}", i, main_backend, register_name);
        OMNI_INSTANCE_REGISTER(
            Backend, register_name, base_dependencies_[i].get());
        reg_backends[base_dependencies_[i].get()] = register_name;
      } else {
        register_name = reg_backends[base_dependencies_[i].get()];
      }

      // todo check illegal name
      find_start = str::replace_once(phases[1], main_backend, register_name);
    }
  }
  forward_backend_ = init_backend(phases[1], config, kwargs);
  OMNI_ASSERT(forward_backend_, "IoCV0 init failed");
  // for (const auto& item : backend_map) {
  //     Backend* backend = OMNI_INSTANCE_GET(Backend, item.first);
  //     OMNI_ASSERT(backend);
  //     backend->inject_dependency(item.second);
  // }
  post_init(config, kwargs);
  SPDLOG_INFO(
      "IoCV0, forward phase: {}, [{}, {}]",
      phases[1],
      forward_backend_->min(),
      forward_backend_->max());
}

void IoCV0::init_phase(
    const std::string& phase_config,
    const std::unordered_map<std::string, std::string>& config,
    const dict& kwargs) {
  SPDLOG_INFO("Ioc init phase: {}", phase_config);
  auto backend_names = str::items_split(phase_config, ',', '[', ']');
  OMNI_ASSERT(
      backend_names.size() >= 1, "Container: backend_names.size() should >= 1");

  std::vector<Backend*> backends;
  for (std::size_t i = 0; i < backend_names.size(); ++i) {
    const auto& engine_name = backend_names[i];

    std::string prefix_str, post_str;
    auto backend =
        str::prefix_parentheses_split(engine_name, prefix_str); // (params1=a)A

    auto pre_config = str::auto_config_split(prefix_str, "filter");
    auto new_config = config;

    // handle A(params1=a)
    backend = str::post_parentheses_split(backend, post_str);
    if (!post_str.empty()) {
      auto post_config = str::auto_config_split(post_str, "key");
      for (auto& [key, value] : post_config) {
        new_config[key] = value;
      }
      SPDLOG_INFO(
          "backend : {} pre: `{}` post: `size={}`",
          engine_name,
          prefix_str,
          new_config.size());
    }
    auto main_backend = str::brackets_split(backend, new_config);
    // OMNI_ASSERT(new_config.find("backend") != new_config.end());
    if (pre_config.find("backend") == pre_config.end())
      pre_config["backend"] = main_backend;
    base_config_.push_back(pre_config);
    auto backend_ptr =
        std::unique_ptr<Backend>(OMNI_CREATE(Backend, main_backend));
    OMNI_ASSERT(backend_ptr, "create " + main_backend + " failed");
    backend_ptr->init(new_config, kwargs);
    backends.push_back(backend_ptr.get());
    base_dependencies_.emplace_back(std::move(backend_ptr));
  }
}
OMNI_REGISTER(Backend, IoCV0);

class IoC : public ControlPlane {
  virtual void impl_custom_init(
      const std::unordered_map<std::string, std::string>& params,
      const dict& options) override {
    backends_.resize(backend_cfgs_.size());
    size_t num_semicolon =
        std::count(delimiters_.begin(), delimiters_.end(), ';');
    OMNI_ASSERT((num_semicolon == 1 && delimiters_.back() == ';'));

    // initialization phase
    for (size_t i = 0; i < backend_cfgs_.size() - 1; ++i) {
      std::unique_ptr<Backend> backend =
          init_backend(backend_cfgs_[i], params, options);
      backends_[i] = std::move(backend);
    }

    // forward phase
    std::unordered_map<void*, std::string> reg_backends;
    std::unordered_set<std::string> main_backends_set(
        main_backends_.begin(), main_backends_.end());
    OMNI_ASSERT(main_backends_set.size() == main_backends_.size());

    auto& forward_backend = backend_cfgs_.back();
    for (size_t i = 0; i < main_backends_.size() - 1; ++i) {
      const auto& main_backend = main_backends_[i];

      size_t find_start = 0;
      std::unordered_map<void*, std::string> reg_backends;
      while (forward_backend.find(main_backend, find_start) !=
             std::string::npos) {
        std::string register_name;
        auto iter_reg = reg_backends.find(backends_[i].get());
        if (iter_reg == reg_backends.end()) {
          register_name = "ioc." + main_backend +
              "." + // std::to_string(i) + "." +
              std::to_string(get_unique_index());
          OMNI_INSTANCE_REGISTER(Backend, register_name, backends_[i].get());
          reg_backends[backends_[i].get()] = register_name;
        } else {
          register_name = iter_reg->second;
        }

        // todo check illegal name
        find_start =
            str::replace_once(forward_backend, main_backend, register_name);
      }
    }
    std::unique_ptr<Backend> backend =
        init_backend(forward_backend, params, options);
    backends_.push_back(std::move(backend));
  }

  void impl_forward(const std::vector<dict>& io) {
    return backends_.back()->forward(io);
  }
  // [[nodiscard]] virtual uint32_t impl_min() const override {
  //     return backends_.back()->min();
  // }
  // [[nodiscard]] virtual uint32_t impl_max() const override {
  //     return backends_.back()->max();
  // };

  void update_min_max() override {
    min_ = backends_.back()->min();
    max_ = backends_.back()->max();
  }

 private:
  // Default class name if the instance is not create via reflection.
  // virtual std::string default_cls_name() const override {
  //   return "IoC";
  // }

  std::vector<std::unique_ptr<Backend>> backends_;

 public:
  ~IoC() {
    // order is important here
    while (!backends_.empty()) {
      backends_.pop_back();
    }
  }
};
OMNI_REGISTER_BACKEND(IoC);

class With : public ControlPlane {
  virtual void impl_custom_init(
      const std::unordered_map<std::string, std::string>& params,
      const dict& options) override {
    backends_.resize(backend_cfgs_.size());
    OMNI_ASSERT(backend_cfgs_.size() == 2);
    size_t num_semicolon =
        std::count(delimiters_.begin(), delimiters_.end(), ',');
    OMNI_ASSERT((num_semicolon == 1 && delimiters_.back() == ','));

    // initialization phase
    for (size_t i = 0; i < backend_cfgs_.size(); ++i) {
      std::unique_ptr<Backend> backend =
          init_backend(backend_cfgs_[i], params, options);
      backends_[i] = std::move(backend);
    }
    OMNI_ASSERT(
        backends_.front()->max() >= backends_.back()->max(),
        std::to_string(backends_.front()->max()) + " Vs. " +
            std::to_string(backends_.back()->max()));
  }

  void impl_forward(const std::vector<dict>& ios) {
    backends_.front()->forward_with_dep(ios, *(backends_.back().get()));
  }

  void update_min_max() override {
    min_ = backends_.back()->min();
    max_ = backends_.back()->max();
  }

 private:
  std::vector<std::unique_ptr<Backend>> backends_;

 public:
  ~With() {
    // order is important here
    while (!backends_.empty()) {
      backends_.pop_back();
    }
  }
};
OMNI_REGISTER_BACKEND(With);
} // namespace omniback