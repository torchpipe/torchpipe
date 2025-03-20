#include <atomic>
#include "hami/helper/unique_index.hpp"

namespace hami {
size_t get_unique_index() {
  static std::atomic<size_t> counter(0);
  return counter.fetch_add(1, std::memory_order_relaxed);
}
}  // namespace hami