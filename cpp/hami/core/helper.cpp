
#include <algorithm>
#include "hami/core/helper.hpp"
#include "hami/core/event.hpp"
#include "hami/core/backend.hpp"

namespace hami {

bool event_guard(Backend* dependency, const std::vector<dict>& inputs) {
  const bool all_have_event = std::all_of(inputs.begin(), inputs.end(), [](const auto& item) {
    return item->find(TASK_EVENT_KEY) != item->end();
  });

  if (all_have_event) {
    return false;
  }
  const bool none_have_event = std::none_of(inputs.begin(), inputs.end(), [](const auto& item) {
    return item->find(TASK_EVENT_KEY) != item->end();
  });

  if (none_have_event) {
    auto ev = make_event(inputs.size());
    for (auto& item : inputs) {
      (*item)[TASK_EVENT_KEY] = ev;
    }
    dependency->forward(inputs);

    auto exc = ev->wait_and_get_except();

    for (auto& item : inputs) {
      item->erase(TASK_EVENT_KEY);
    }

    if (exc) {
      std::rethrow_exception(exc);
    }
  } else {
    throw std::logic_error(
        "event_guard: Inconsistent event state in inputs. All inputs should be either async or "
        "sync.");
  }
  return true;
}

}  // namespace hami