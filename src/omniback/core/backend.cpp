
#include "omniback/core/backend.hpp"
#include <memory>
#include "omniback/core/event.hpp"
#include "omniback/core/helper.hpp"
#include "omniback/core/parser.hpp"
#include "omniback/core/reflect.h"
#include "omniback/helper/base_logging.hpp"
#include "omniback/helper/string.hpp"

namespace omniback {

// void Backend::get_class_name(std::string& default_name) const {
//     auto name = OMNI_OBJECT_NAME(Backend, this);
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
void HasEventForwardGuard::impl_forward(const std::vector<dict>& inputs) {
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
    auto ev = Event(inputs.size());
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

std::unique_ptr<Backend> create_backend(
    const std::string& class_name,
    const std::string& aspect_name_str) {
  auto backend = std::unique_ptr<Backend>(
      OMNI_CREATE(Backend, class_name, aspect_name_str));
  OMNI_ASSERT(
      backend != nullptr,
      "Failed to create backend `" + class_name + "` through reflection");
  return backend;
};

void register_backend(
    const std::string& aspect_name_str,
    std::shared_ptr<Backend> backend) {
  OMNI_INSTANCE_REGISTER(Backend, aspect_name_str, backend);
}
void unregister_backend(const std::string& aspect_name_str) {
  OMNI_INSTANCE_UNREGISTER(Backend, aspect_name_str);
}

void cleanup_backend() {
  OMNI_INSTANCE_CLEANUP(Backend);
  OMNI_CLEANUP_ALL(Backend);
}

OMNI_REGISTER(Backend, Backend, "Backend, Pass");

// std::unique_ptr<Backend> init_backend(
//     const std::string &backend_config,
//     std::unordered_map<std::string, std::string> dst_config, const dict
//     &kwargs, const std::string &aspect_name_str)
// {
//     auto main_backend = str::brackets_split(backend_config, dst_config);
//     auto backend = std::unique_ptr<Backend>(
//         OMNI_CREATE(Backend, main_backend, aspect_name_str));
//     OMNI_ASSERT(backend != nullptr, "Failed to create backend " +
//     main_backend +
//                                         " through reflection");
//     backend->init(dst_config, kwargs);
//     return backend;
// };

std::unique_ptr<Backend> init_backend(
    const std::string& backend_config,
    std::unordered_map<std::string, std::string> dst_config,
    const dict& dict_kwargs,
    const std::string& aspect_name_str) {
  parser_v2::Parser parser;
  std::string main_backend = parser.parse(backend_config, dst_config);

  auto backend = std::unique_ptr<Backend>(
      OMNI_CREATE(Backend, main_backend, aspect_name_str));
  OMNI_ASSERT(
      backend != nullptr,
      "Failed to create backend " + main_backend + " through reflection");
  backend->init(dst_config, dict_kwargs);
  return backend;
};

Backend* get_backend(const std::string& aspect_name_str) {
  return OMNI_INSTANCE_GET(Backend, aspect_name_str);
}

#ifdef DEBUG
// void Backend::forward(const std::vector<dict>& io) {
//   std::optional<std::string> node_name;
//   auto cls_name = OMNI_OBJECT_NAME(Backend, this);
//   if (!io.empty()) {
//     node_name = try_get<std::string>(io[0], "node_name");
//   }
//   SPDLOG_INFO("tracing: node={}, cls_name={}", *node_name, *cls_name);
//   impl_forward(io);
// }
#endif

void Backend::safe_forward(const std::vector<dict>& input_output) {
  size_t io_size = get_request_size(input_output);
  if (io_size >= min() && io_size <= max()) {
    forward(input_output);
  } else if (1 == max()) { // special case
    for (const auto& item : input_output) {
      forward({item});
    }
  } else if (input_output.size() > max()) {
    forward(
        std::vector<dict>(input_output.begin(), input_output.begin() + max()));
    const auto& left =
        std::vector<dict>(input_output.begin() + max(), input_output.end());
    if (!left.empty())
      safe_forward(left);
  } else if (input_output.size() < min()) {
    throw std::invalid_argument("input_output.size() < min()");
  }
}

void BackendMax::impl_forward(const std::vector<dict>& input_output) {
  for (const auto& item : input_output) {
    (*item)[TASK_RESULT_KEY] = item->at(TASK_DATA_KEY);
  }
}
OMNI_REGISTER_BACKEND(BackendMax);

namespace backend {
void evented_forward(Backend& self, const std::vector<dict>& inputs) {
  const bool all_have_event =
      std::all_of(inputs.begin(), inputs.end(), [](const auto& item) {
        return item->find(TASK_EVENT_KEY) != item->end();
      });

  if (all_have_event) {
    self.forward(inputs);
    return;
  }
  const bool none_have_event =
      std::none_of(inputs.begin(), inputs.end(), [](const auto& item) {
        return item->find(TASK_EVENT_KEY) != item->end();
      });

  if (none_have_event) {
    auto ev = Event(inputs.size());
    for (auto& item : inputs) {
      (*item)[TASK_EVENT_KEY] = ev;
    }
    self.forward(inputs);

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

std::string get_dependency_name(
    const Backend* this_ptr,
    const std::unordered_map<std::string, std::string>& config,
    const std::optional<std::string>& defualt_cls_name) {
  auto name = OMNI_OBJECT_NAME(Backend, this_ptr);
  if (!name)
    name = defualt_cls_name;
  OMNI_ASSERT(name, "this instance was not created via reflection");
  auto iter = config.find(*name + "::dependency");
  OMNI_ASSERT(
      iter != config.end(),
      *name + "::dependency" + " not found in configuration. ");
  return iter->second;
}
} // namespace backend
} // namespace omniback

namespace omniback::parser_v2 {
bool get_backend_name(const Backend* obj_ptr, std::string& cls_name) {
  auto name = OMNI_OBJECT_NAME(Backend, obj_ptr);
  if (name) {
    cls_name = *name;
    return true;
  }
  return false;
}

std::string get_backend_name(const Backend* obj_ptr) {
  auto name = OMNI_OBJECT_NAME(Backend, obj_ptr);
  OMNI_ASSERT(name, "this instance was not created via reflection");
  return *name;
}

std::optional<std::string> get_opt_dependency_name(
    const Backend* this_ptr,
    const std::unordered_map<std::string, std::string>& config) {
  auto name = OMNI_OBJECT_NAME(Backend, this_ptr);

  OMNI_ASSERT(name, "this instance was not created via reflection");
  auto iter = config.find(*name + "::dependency");
  if (iter == config.end())
    return std::nullopt;
  return iter->second;
}

std::string get_dependency_name(
    const Backend* this_ptr,
    const std::unordered_map<std::string, std::string>& config,
    const std::optional<std::string>& defualt_cls_name) {
  auto name = OMNI_OBJECT_NAME(Backend, this_ptr);
  if (!name)
    name = defualt_cls_name;
  OMNI_ASSERT(name, "this instance was not created via reflection");
  auto iter = config.find(*name + "::dependency");
  OMNI_ASSERT(
      iter != config.end(),
      *name + "::dependency" + " not found in configuration. ");
  return iter->second;
}

ArgsKwargs get_args_kwargs(
    const Backend* obj_ptr,
    std::string cls_name,
    const std::unordered_map<std::string, std::string>& config) {
  auto name = OMNI_OBJECT_NAME(Backend, obj_ptr);
  if (name) {
    cls_name = *name;
  } else {
    OMNI_ASSERT(
        !cls_name.empty(), "This instance was not created via reflection");
  }
  auto iter = config.find(cls_name + "::args");
  if (iter != config.end()) {
    auto [args, str_kwargs] = parser::parse_args_kwargs(iter->second);
    parser::update(config, str_kwargs);
    return {args, str_kwargs};
  }

  return {{}, config};
}
} // namespace omniback::parser_v2