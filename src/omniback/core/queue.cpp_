

#include "omniback/core/queue.hpp"

namespace omniback {

Queue& default_src_queue() {
  static Queue result_queue;
  return result_queue;
}

Queue& default_output_queue() {
  static Queue result_queue;
  return result_queue;
}

Queue& default_queue(const std::string& tag) {
  static std::mutex mtx;
  static std::unordered_map<std::string, Queue> queue_map;

  std::lock_guard<std::mutex> lock(mtx);
  return queue_map[tag];
}

// SizedQueue& default_sized_queue() {
//     static SizedQueue result_queue;
//     return result_queue;
// }
} // namespace omniback