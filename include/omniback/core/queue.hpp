#pragma once

// #include "omniback/core/dict.hpp"
// #include "omniback/core/general_queue.hpp"
#include "omniback/ffi/queue.h"


namespace omniback {
// using ffi::Queue;
// using ffi::default_queue;
using Queue = ffi::ThreadSafeQueueObj;

using ffi::default_queue;
// using LegacyQueue = queue::ThreadSafeSizedQueue<dict>;
// using LegacySizedQueue = queue::ThreadSafeSizedQueue<dict>;
// // Queue& default_queue();

// Queue& default_src_queue();
// Queue& default_output_queue();

// Queue& default_queue(const std::string& tag = "");

// SizedQueue& default_sized_queue();
} // namespace omniback
