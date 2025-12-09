#pragma once

#include <condition_variable>
#include <string>
#include <thread>
#include <vector>
#include "omniback/core/backend.hpp"
#include "omniback/core/queue.hpp"

namespace omniback {
class Source : public Backend {
  void impl_init(
      const std::unordered_map<std::string, std::string>& config,
      const dict& kwargs) override;
  void impl_forward(const std::vector<dict>& input) override;

 public:
  // ~Source() { bInited_.store(false); }

 private:
  std::atomic_bool bInited_{false};
  size_t total_number_ = 0;
};

dict uniform_sample(const std::vector<dict>& input);
} // namespace omniback