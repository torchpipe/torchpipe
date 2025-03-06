
#include "hami/helper/macro.h"
#include "hami/core/task_keys.hpp"
#include "hami/core/reflect.h"
#include "hami/helper/base_logging.hpp"
#include "hami/helper/string.hpp"

#include "hami/builtin/ioc.hpp"
#include "hami/helper/unique_index.hpp"
namespace hami {

void IoC::init(const std::unordered_map<std::string, std::string>& in_config,
               const dict& in_dict_config) {
    constexpr auto default_name = "IoC";
    auto name = HAMI_OBJECT_NAME(Backend, this);
    if (!name) {
        name = default_name;
        SPDLOG_WARN(
            "{}::init, instance not created via reflection, using default "
            "name {}. "
            "Configure dependencies via {}::dependency",
            *name, *name, *name);
    }
    auto config = in_config;
    auto iter = config.find(*name + "::dependency");
    HAMI_ASSERT(iter != config.end(),
                "Dependency configuration missing for " + *name);

    const auto& backend_setting = iter->second;
    config.erase(iter);

    std::vector<std::string> phases = str::items_split(backend_setting, ';');
    HAMI_ASSERT(phases.size() == 2, "IoC requires two phases separated by ';'");

    auto dict_config = in_dict_config ? in_dict_config : make_dict();
    init_phase(phases[0], config, dict_config);  // Initialization phase

    // std::unordered_map<std::string, Backend*> backend_map;
    std::unordered_set<std::string> keys;
    for (size_t i = 0; i < base_config_.size(); ++i) {
        const auto& item = base_config_[i];
        auto main_backend = item.at("backend");
        HAMI_ASSERT(
            keys.count(main_backend) == 0,
            "Duplicate backend name detected during initialization parsing: " +
                main_backend);
        keys.insert(main_backend);
        size_t find_start = 0;
        std::unordered_map<void*, std::string> registered_backends;
        while (phases[1].find(main_backend, find_start) != std::string::npos) {
            std::string register_name;
            if (registered_backends.find(base_dependencies_[i].get()) ==
                registered_backends.end()) {
                register_name = "ioc.proxy." + main_backend +
                                "." +  // std::to_string(i) + "." +
                                std::to_string(get_unique_index());
                HAMI_INSTANCE_REGISTER(Backend, register_name,
                                       base_dependencies_[i].get());
                registered_backends[base_dependencies_[i].get()] =
                    register_name;
            } else {
                register_name =
                    registered_backends[base_dependencies_[i].get()];
            }

            // todo check illegal name
            find_start =
                str::replace_once(phases[1], main_backend, register_name);
        }
    }
    forward_backend_ = init_backend(phases[1], config, dict_config);
    HAMI_ASSERT(forward_backend_, "IoC init failed");
    // for (const auto& item : backend_map) {
    //     Backend* backend = HAMI_INSTANCE_GET(Backend, item.first);
    //     HAMI_ASSERT(backend);
    //     backend->inject_dependency(item.second);
    // }
    post_init(config, dict_config);
}

[[nodiscard]] size_t IoC::max() const { forward_backend_->max(); }
[[nodiscard]] size_t IoC::min() const { forward_backend_->min(); }

void IoC::init_phase(const std::string& phase_config,
                     const std::unordered_map<std::string, std::string>& config,
                     const dict& dict_config) {
    auto backend_names = str::items_split(phase_config, ',', '[', ']');
    HAMI_ASSERT(backend_names.size() >= 1,
                "Container: backend_names.size() should >= 1");

    std::vector<Backend*> backends;
    for (std::size_t i = 0; i < backend_names.size(); ++i) {
        const auto& engine_name = backend_names[i];

        std::string prefix_str, post_str;
        auto backend = str::prefix_parentheses_split(
            engine_name, prefix_str);  // (params1=a)A

        auto pre_config = str::auto_config_split(prefix_str, "filter");
        auto new_config = config;

        // handle A(params1=a)
        backend = str::post_parentheses_split(backend, post_str);
        if (!post_str.empty()) {
            auto post_config = str::auto_config_split(post_str, "key");
            for (auto& [key, value] : post_config) {
                new_config[key] = value;
            }
            SPDLOG_INFO("backend : {} pre: `{}` post: `size={}`", engine_name,
                        prefix_str, new_config.size());
        }
        auto main_backend = str::brackets_split(backend, new_config);
        // HAMI_ASSERT(new_config.find("backend") != new_config.end());
        if (pre_config.find("backend") == pre_config.end())
            pre_config["backend"] = main_backend;
        base_config_.push_back(pre_config);
        auto backend_ptr =
            std::unique_ptr<Backend>(HAMI_CREATE(Backend, main_backend));
        HAMI_ASSERT(backend_ptr, "create " + main_backend + " failed");
        backend_ptr->init(new_config, dict_config);
        backends.push_back(backend_ptr.get());
        base_dependencies_.emplace_back(std::move(backend_ptr));
    }
}
HAMI_REGISTER(Backend, IoC);
}  // namespace hami