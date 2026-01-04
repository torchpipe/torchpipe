#include <random>

#include "omniback/builtin/source.hpp"
#include "omniback/helper/timer.hpp"

namespace omniback {
void Source::impl_init(
    const std::unordered_map<std::string, std::string>& config,
    const dict& kwargs) {
  str::try_update<size_t>(config, "max_number", total_number_);
  // OMNI_ASSERT(total_number_ > 0);
}
dict uniform_sample(const std::vector<dict>& input) {
  thread_local std::random_device seeder;
  thread_local std::mt19937 engine(seeder());
  thread_local std::uniform_int_distribution<int> dist(0, input.size() - 1);
  return copy_dict(input[dist(engine)]);
}

void Source::impl_forward(const std::vector<dict>& input) {
  std::random_device seeder;
  std::mt19937 engine(seeder());
  std::uniform_int_distribution<int> dist(0, input.size() - 1);
  auto src_queue = &default_queue("source");

  if (total_number_ > 0) {
    size_t num_finish = 0;
    while (num_finish < total_number_) {
      if (src_queue->push_with_max_limit(
              copy_dict(input[dist(engine)]),
              20,
              SHUTDOWN_TIMEOUT)) {
        num_finish++;
      }
    }
  } else {
    src_queue->pushes(input);
  }
}

OMNI_REGISTER_BACKEND(Source, "uniformSample,Source");
} // namespace omniback