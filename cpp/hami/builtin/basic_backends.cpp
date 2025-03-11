
#include <memory>
#include <numeric>
#include "hami/core/reflect.h"
#include "hami/helper/base_logging.hpp"
#include "hami/helper/string.hpp"
#include "hami/builtin/basic_backends.hpp"
#include "hami/helper/macro.h"
#include "hami/core/helper.hpp"

namespace hami {
void Dependency::init(
    const std::unordered_map<std::string, std::string>& config,
    const dict& dict_config) {
    HAMI_ASSERT(!shared_owned_dependency_, "Duplicate initialization");
    pre_init(config, dict_config);

    if (dependency_name_.empty()) {
        auto dep = get_dependency_name(this, config);
        if (dep.has_value()) {
            dependency_name_ = *dep;
        }
    }
    if (!dependency_name_.empty()) {
        auto backend =
            std::shared_ptr<Backend>(HAMI_CREATE(Backend, dependency_name_));
        HAMI_ASSERT(backend,
                    "`" + dependency_name_ + "` is not a valid backend");
        backend->init(config, dict_config);
        {
            if (!registered_name_.empty()) {
                HAMI_INSTANCE_REGISTER(Backend, registered_name_, backend);
            }
            shared_owned_dependency_ = backend;
        }
        inject_dependency(backend.get());
    } else {
        SPDLOG_DEBUG(
            "*::dependency not found, skipping "
            "dependency injection process");
    }
    SPDLOG_INFO("Dependency dependency_name_ = {}", dependency_name_);
    post_init(config, dict_config);
}

void Dependency::set_dependency_name(
    const std::unordered_map<std::string, std::string>& config,
    const std::string& default_cls_name, const std::string& default_dep_name) {
    dependency_name_ =
        get_dependency_name(this, config, default_cls_name, default_dep_name);
}

void Dependency::inject_dependency(Backend* dependency) {
    if (dependency == nullptr) {
        throw std::invalid_argument("null dependency is not allowed");
    }
    if (injected_dependency_) {
        [[maybe_unused]] thread_local const auto log_tmp = []() {
            SPDLOG_WARN(
                "Dependency::inject_dependency: dependency already exists(may "
                "happened in the "
                "pre_init stage). Chain dependency injection "
                "will be applied.");
            return 0;
        }();
        injected_dependency_->inject_dependency(dependency);
    } else {
        injected_dependency_ = dependency;
    }
}
void Dependency::forward_dep_impl(const std::vector<dict>& input_output,
                                  Backend* dependency) {
    dependency->safe_forward(input_output);
}

Dependency::~Dependency() {
    if (!registered_name_.empty()) {
        HAMI_INSTANCE_UNREGISTER(Backend, registered_name_);
    }
}

void DynamicDependency::inject_dependency(Backend* dependency) {
    HAMI_ASSERT(dependency && !injected_dependency_);

    injected_dependency_ = dependency;
}
size_t DynamicDependency::max() const {
    HAMI_ASSERT(injected_dependency_,
                "Dependency not initialized. Call inject_dependency first.");
    return injected_dependency_->max();
}

size_t DynamicDependency::min() const {
    HAMI_ASSERT(injected_dependency_,
                "Dependency not initialized. Call inject_dependency first.");
    return injected_dependency_->min();
}

void Container::init(const std::unordered_map<std::string, std::string>& config,
                     const dict& dict_config) {
    constexpr auto default_name = "Container";
    auto name = HAMI_OBJECT_NAME(Backend, this);
    if (name == std::nullopt) {
        name = default_name;
        SPDLOG_WARN(
            "{}::init, it seems this instance was not created via reflection, "
            "using default name {}. "
            "Please configure its dependency via the parameter {}::dependency",
            *name, *name, *name);
    }
    auto iter = config.find(*name + "::dependency");
    HAMI_ASSERT(iter != config.end(),
                "Dependency configuration " + *name +
                    "::dependency not found. "
                    "Containers do not allow runtime dynamic modification of "
                    "dependencies, "
                    "please specify dependencies in the configuration");

    const auto& backend_setting = iter->second;
    // SPDLOG_DEBUG("Expand container to {}[{}].", name, backend_setting);

    if (!backend_setting.empty()) {
        auto backend_names = str::items_split(backend_setting, ',', '[', ']');
        HAMI_ASSERT(backend_names.size() >= 1,
                    "Container: backend_names.size() should >= 1");

        auto order = set_init_order(backend_names.size());
        base_config_.resize(backend_names.size());
        base_dependencies_.resize(backend_names.size());

        bool lazy_init = order.size() != backend_names.size();

        // std::reverse(backend_names.begin(), backend_names.end());
        std::vector<Backend*> backends(backend_names.size());
        for (std::size_t index = 0; index < backend_names.size(); ++index) {
            auto i = lazy_init ? index : order[index];
            const auto& engine_name = backend_names[i];

            std::string prefix_str, post_str;
            auto backend = str::prefix_parentheses_split(
                engine_name, prefix_str);  // (params1=a)A

            auto pre_config = str::auto_config_split(prefix_str, "filter");
            auto new_config = config;
            new_config.erase(*name + "::dependency");

            // handle A(params1=a)
            backend = str::post_parentheses_split(backend, post_str);
            if (!post_str.empty()) {
                auto post_config = str::auto_config_split(post_str, "key");
                for (auto& [key, value] : post_config) {
                    new_config[key] = value;
                }
                SPDLOG_INFO("backend : {} pre: `{}` post: `size={}`",
                            engine_name, prefix_str, new_config.size());
            }
            auto main_backend = str::brackets_split(backend, new_config);
            // HAMI_ASSERT(new_config.find("backend") != new_config.end());
            if (pre_config.find("backend") == pre_config.end())
                pre_config["backend"] = main_backend;
            backends_.push_back(main_backend);
            base_config_[i] = (pre_config);
            auto backend_ptr =
                std::unique_ptr<Backend>(HAMI_CREATE(Backend, main_backend));
            HAMI_ASSERT(
                backend_ptr,
                "create " + main_backend +
                    " failed. This is not a Backend name. May be this is an "
                    "instance name and you forgot to use it with Forward[*]");
            if (lazy_init) {
                auto* pbackend_ptr = backend_ptr.get();
                lazy_init_func_.emplace_back(
                    [new_config, dict_config, pbackend_ptr]() {
                        pbackend_ptr->init(new_config, dict_config);
                    });
            } else {
                backend_ptr->init(new_config, dict_config);
            }

            backends[i] = (backend_ptr.get());
            base_dependencies_[i] = std::move(backend_ptr);
        }

        // std::reverse(base_dependencies_.begin(), base_dependencies_.end());
        // std::reverse(backends.begin(), backends.end());
        // std::reverse(base_config_.begin(), base_config_.end());
        if (!lazy_init) {
            auto [min_value, max_value] = update_min_max(backends);
            min_ = min_value;
            max_ = max_value;
        }
    } else {
        HAMI_THROW("Wired. Empty config.");
    }
    post_init(config, dict_config);
}

std::vector<size_t> Container::set_init_order(size_t max_range) const {
    std::vector<size_t> order(max_range, 0);
    std::iota(order.rbegin(), order.rend(), 0);
    return order;
}

std::pair<size_t, size_t> Container::update_min_max(
    const std::vector<Backend*>& depends) {
    size_t max_value = std::numeric_limits<size_t>::max();
    size_t min_value = 1;
    size_t num_one = 0;
    for (Backend* depend : depends) {
        if (depend->max() == 1) {
            num_one++;
        } else {
            min_value = std::max(min_value, depend->min());
            max_value = std::min(max_value, depend->max());
        }
    }

    if (num_one == depends.size()) {
        max_value = 1;
    }
    HAMI_ASSERT(min_value <= max_value);
    // min_ = min_value;
    // max_ = max_value;
    return {min_value, max_value};
}

void List::init(const std::unordered_map<std::string, std::string>& config,
                const dict& dict_config) {
    constexpr auto default_name = "List";
    auto name = HAMI_OBJECT_NAME(Backend, this);
    if (name == std::nullopt) {
        name = default_name;
        SPDLOG_WARN(
            "{}::init, it seems this instance was not created via reflection, "
            "using default name {}. "
            "Please configure its dependency via the parameter {}::dependency",
            *name, *name, *name);
    }
    auto iter = config.find(*name + "::dependency");
    HAMI_ASSERT(iter != config.end(),
                "Dependency configuration " + *name +
                    "::dependency not found. "
                    "Containers do not allow runtime dynamic modification of "
                    "dependencies, "
                    "please specify dependencies in the configuration");

    const auto& backend_setting = iter->second;
    // SPDLOG_DEBUG("Expand container to {}[{}].", name, backend_setting);

    if (!backend_setting.empty()) {
        auto backend_names = str::items_split(backend_setting, ',', '[', ']');
        HAMI_ASSERT(backend_names.size() >= 1,
                    "Container: backend_names.size() should >= 1");

        // std::reverse(backend_names.begin(), backend_names.end());

        for (std::size_t i = 0; i < backend_names.size(); ++i) {
            const auto& engine_name = backend_names[i];

            std::string prefix_str, post_str;
            auto backend = str::prefix_parentheses_split(
                engine_name, prefix_str);  // (params1=a)A

            auto pre_config = str::auto_config_split(prefix_str, "filter");
            auto new_config = config;
            new_config.erase(*name + "::dependency");

            // handle A(params1=a)
            backend = str::post_parentheses_split(backend, post_str);
            if (!post_str.empty()) {
                auto post_config = str::auto_config_split(post_str, "key");
                for (auto& [key, value] : post_config) {
                    new_config[key] = value;
                }
                SPDLOG_INFO("backend : {} pre: `{}` post: `size={}`",
                            engine_name, prefix_str, new_config.size());
            }
            auto main_backend = str::brackets_split(backend, new_config);
            // HAMI_ASSERT(new_config.find("backend") != new_config.end());
            if (pre_config.find("backend") == pre_config.end())
                pre_config["backend"] = main_backend;
            // base_config_.push_back(pre_config);
            auto backend_ptr =
                std::unique_ptr<Backend>(HAMI_CREATE(Backend, main_backend));
            HAMI_ASSERT(backend_ptr, "create " + main_backend + " failed");
            backend_ptr->init(new_config, dict_config);
            backends_.push_back(std::move(backend_ptr));
            // base_dependencies_.emplace_back(std::move(backend_ptr));
        }

    } else {
        HAMI_THROW("Wired. Empty config.");
    }
}

void List::forward(const std::vector<dict>& input_output) {
    throw std::runtime_error("List::forward not implemented");
}

HAMI_REGISTER(Backend, List, "List,Tuple");

}  // namespace hami
