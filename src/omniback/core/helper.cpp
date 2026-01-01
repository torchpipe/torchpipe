
#include "omniback/core/helper.hpp"
#include <algorithm>
#include "omniback/core/backend.hpp"
#include "omniback/core/event.hpp"
#include "omniback/helper/base_logging.hpp"

namespace omniback {

DictHelper& DictHelper::keep(const std::string& key) {
  std::vector<std::optional<any>> keeped;
  for (const auto& da : dicts_) {
    OMNI_ASSERT(da->find(TASK_EVENT_KEY) == da->end());
    auto iter = da->find(key);
    if (iter == da->end()) {
      keeped.emplace_back(std::nullopt);
    } else {
      keeped.emplace_back(iter->second);
    }
  }
  keep_[key] = keeped;
  return *this;
}

HasEventHelper::HasEventHelper(const std::vector<dict>& data) : dicts_(data) {
  const bool all_have_event =
      std::all_of(data.begin(), data.end(), [](const auto& item) {
        return item->find(TASK_EVENT_KEY) != item->end();
      });

  if (all_have_event) {
    // SPDLOG_INFO("HasEventHelper: all_have_event");
    return;
  }
  const bool none_have_event =
      std::none_of(data.begin(), data.end(), [](const auto& item) {
        return item->find(TASK_EVENT_KEY) != item->end();
      });

  if (none_have_event) {
    // SPDLOG_INFO(
    //     "HasEventHelper: none_have_event. data.size() = {}", data.size());
    event_ = Event(data.size());
    for (auto& item : data) {
      (*item)[TASK_EVENT_KEY] = event_.value();
    }
  } else {
    throw std::logic_error(
        "HasEventHelper: Inconsistent event state in inputs. All "
        "inputs "
        "should be either async or "
        "sync.");
  }
  return;
}

void HasEventHelper::wait() {
  if (event_.has_value()) {
    event_.value()->wait_finish();
    for (std::size_t i = 0; i < dicts_.size(); ++i) {
      dicts_[i]->erase(TASK_EVENT_KEY);
    }
    std::optional<Event> tmp;
    std::swap(tmp, event_);
    tmp.value()->try_throw();
  }
}

HasEventHelper::~HasEventHelper() {
  if (event_.has_value()) {
    SPDLOG_ERROR("HasEventHelper: event not cleared. call wait()");
    std::terminate();
  }
}

void event_guard_forward(
    std::function<void(const std::vector<dict>&)> func,
    const std::vector<dict>& inputs) {
  const bool all_have_event =
      std::all_of(inputs.begin(), inputs.end(), [](const auto& item) {
        return item->find(TASK_EVENT_KEY) != item->end();
      });

  if (all_have_event) {
    func(inputs);
    // return false;
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
    func(inputs);

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
  // return true;
}

std::string get_dependency_name_force(
    const Backend* this_ptr,
    const std::unordered_map<std::string, std::string>& config) {
  auto name = OMNI_OBJECT_NAME(Backend, this_ptr);
  OMNI_ASSERT(
      name != std::nullopt, "this instance was not created via reflection");
  auto iter = config.find(*name + "::dependency");
  OMNI_ASSERT(iter != config.end(), *name + "::dependency" + " not found. ");
  return iter->second;
}

std::optional<std::string> get_dependency_name(
    const Backend* this_ptr,
    const std::unordered_map<std::string, std::string>& config) {
  auto name = OMNI_OBJECT_NAME(Backend, this_ptr);
  if (name == std::nullopt) {
    throw std::runtime_error("This instance was not created via reflection");
  }

  auto iter = config.find(*name + "::dependency");
  if (iter == config.end())
    return std::nullopt;
  return iter->second;
}

std::optional<std::string> get_dependency_name(
    const Backend* this_ptr,
    const std::unordered_map<std::string, std::string>& config,
    const std::string& default_name) {
  auto name = OMNI_OBJECT_NAME(Backend, this_ptr);
  if (name == std::nullopt) {
    name = default_name;
    SPDLOG_WARN(
        "{}::init, it seems this instance was not created via reflection, "
        "using default name {}. "
        "Please configure its dependency via the parameter {}::dependency",
        *name,
        *name,
        *name);
  }
  auto iter = config.find(*name + "::dependency");
  if (iter == config.end()) {
    SPDLOG_INFO(
        "Dependency configuration " + *name +
        "::dependency not found. "
        "please specify dependencies in the configuration through " +
        *name + "[X] or {" + *name + "::dependency, X} or do it manually.");
    return std::nullopt;
  }

  return iter->second;
}

std::string get_dependency_name(
    const Backend* this_ptr,
    const std::unordered_map<std::string, std::string>& config,
    const std::string& default_cls_name,
    const std::string& default_dep_name) {
  auto name = get_dependency_name(this_ptr, config, default_cls_name);
  if (name == std::nullopt) {
    return default_dep_name;
  }
  return *name;
}

std::string parse_dependency_from_param(
    const Backend* this_ptr,
    std::unordered_map<std::string, std::string>& config,
    std::string default_params_name,
    const std::string& default_dep_name) {
  auto name = OMNI_OBJECT_NAME(Backend, this_ptr);
  if (name == std::nullopt) {
    throw std::runtime_error("This instance was not created via reflection");
  }
  auto iter = config.find(*name + "::dependency");
  if (iter == config.end()) {
    if (default_params_name.empty()) {
      throw std::invalid_argument(
          "Dependency configuration " + *name +
          "::dependency not found. Please specify dependencies in the "
          "configuration "
          "through " +
          *name + "[X] or {" + *name + "::dependency, X}.");
    }

  } else {
    default_params_name = iter->second;
    config.erase(iter);
  }
  auto params = str::str_split(default_params_name, ',');
  OMNI_ASSERT(params.size() >= 1, "error params: " + default_params_name);
  iter = config.find(params[0]);
  if (iter == config.end()) {
    if (params.size() > 1) {
      return params[1];
    } else {
      OMNI_ASSERT(
          !default_dep_name.empty(),
          "In config, cannot find key `" + default_params_name +
              "`, please check the configuration");
      // SPDLOG_INFO("Using defalut dependency {}", default_dep_name);

      return default_dep_name;
    }
  }

  return iter->second;
}

void notify_event(const std::vector<dict>& io) {
  for (const auto& item : io) {
    auto iter = item->find(TASK_EVENT_KEY);
    if (iter != item->end()) {
      auto ev = any_cast<Event>(iter->second);
      // SPDLOG_INFO("event notified before");
      ev->notify_all();
      // SPDLOG_INFO("event notified");
    }
  }
}

std::string get_cls_name(
    const Backend* this_ptr,
    const std::string& default_cls_name) {
  auto name = OMNI_OBJECT_NAME(Backend, this_ptr);
  if (name == std::nullopt) {
    return default_cls_name;
  }
  return *name;
}

namespace helper {} // namespace helper

} // namespace omniback