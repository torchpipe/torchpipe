#pragma once

#include "hami/core/queue.hpp"

namespace hami {

Queue& default_queue() {
    static Queue result_queue;
    return result_queue;
}

// SizedQueue& default_sized_queue() {
//     static SizedQueue result_queue;
//     return result_queue;
// }
}  // namespace hami