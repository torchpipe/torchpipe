
#include <memory>
#include "hami/core/backend.hpp"
#include "hami/core/reflect.h"
#include "hami/helper/base_logging.hpp"
#include "hami/helper/string.hpp"
#include "hami/core/event.hpp"

namespace hami {

// void Backend::get_class_name(std::string& default_name) const {
//     auto name = HAMI_OBJECT_NAME(Backend, this);
//     if (name == std::nullopt) {
//         name = default_name;
//         SPDLOG_WARN(
//             "{}::init, it seems this instance was not created via reflection,
//             " "using default name {}. " "Please configure its dependency via
//             the parameter {}::dependency", default_name, default_name,
//             default_name);
//     } else
//         default_name = *name;
// }
void EventBackend::forward(const std::vector<dict>& inputs) {
    const bool all_have_event =
        std::all_of(inputs.begin(), inputs.end(), [](const auto& item) {
            return item->find(TASK_EVENT_KEY) != item->end();
        });

    if (all_have_event) {
        evented_forward(inputs);
        return;
    }
    const bool none_have_event =
        std::none_of(inputs.begin(), inputs.end(), [](const auto& item) {
            return item->find(TASK_EVENT_KEY) != item->end();
        });

    if (none_have_event) {
        auto ev = make_event(inputs.size());
        for (auto& item : inputs) {
            (*item)[TASK_EVENT_KEY] = ev;
        }
        evented_forward(inputs);

        auto exc = ev->wait_and_get_except();

        for (auto& item : inputs) {
            item->erase(TASK_EVENT_KEY);
        }

        if (exc) {
            std::rethrow_exception(exc);
        }
    } else {
        throw std::logic_error(
            "event_guard: Inconsistent event state in inputs. All inputs "
            "should be either async or "
            "sync.");
    }
    return;
}

std::unique_ptr<Backend> create_backend(const std::string& class_name,
                                        const std::string& aspect_name_str) {
    auto backend = std::unique_ptr<Backend>(
        HAMI_CREATE(Backend, class_name, aspect_name_str));
    HAMI_ASSERT(backend != nullptr, "Failed to create backend " + class_name +
                                        " through reflection");
    return backend;
};

void register_backend(const std::string& aspect_name_str,
                      std::shared_ptr<Backend> backend) {
    HAMI_INSTANCE_REGISTER(Backend, aspect_name_str, backend);
}
void unregister_backend(const std::string& aspect_name_str) {
    HAMI_INSTANCE_UNREGISTER(Backend, aspect_name_str);
}

void clearup_backend() { HAMI_INSTANCE_CLEANUP(Backend); }

HAMI_REGISTER(Backend, Backend, "Backend, Pass");

std::unique_ptr<Backend> init_backend(
    const std::string& backend_config,
    std::unordered_map<std::string, std::string> dst_config,
    const dict& dict_config, const std::string& aspect_name_str) {
    auto main_backend = str::brackets_split(backend_config, dst_config);
    auto backend = std::unique_ptr<Backend>(
        HAMI_CREATE(Backend, main_backend, aspect_name_str));
    HAMI_ASSERT(backend != nullptr, "Failed to create backend " + main_backend +
                                        " through reflection");
    backend->init(dst_config, dict_config);
    return backend;
};

Backend* get_backend(const std::string& aspect_name_str) {
    return HAMI_INSTANCE_GET(Backend, aspect_name_str);
}

}  // namespace hami
