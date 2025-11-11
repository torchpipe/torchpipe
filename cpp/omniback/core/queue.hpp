#pragma once

#include "omniback/core/dict.hpp"
#include "omniback/core/general_queue.hpp"

namespace omniback {
using Queue = queue::ThreadSafeSizedQueue<dict>;
// using SizedQueue = queue::ThreadSafeSizedQueue<dict>;
// Queue& default_queue();

Queue& default_src_queue();
Queue& default_output_queue();

Queue& default_queue(const std::string& tag = "");

// SizedQueue& default_sized_queue();
} // namespace omniback
