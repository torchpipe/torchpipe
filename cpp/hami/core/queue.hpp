#pragma once

#include "hami/helper/queue.hpp"
#include "hami/core/dict.hpp"

namespace hami {
using Queue = queue::ThreadSafeQueue<dict>;
// using SizedQueue = queue::ThreadSafeSizedQueue<dict>;
Queue& default_queue();
// SizedQueue& default_sized_queue();
}  // namespace hami