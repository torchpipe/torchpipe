#pragma once

#include <string>
#include <vector>
#include <thread>
#include <condition_variable>
#include "hami/core/queue.hpp"
#include "hami/core/backend.hpp"

namespace hami {
class Source : public Backend {
    void impl_init(const std::unordered_map<std::string, std::string>& config,
                   const dict& kwargs) override;
    void impl_forward(const std::vector<dict>& input) override;

   public:
    // ~Source() { bInited_.store(false); }

   private:
    std::atomic_bool bInited_{false};
    size_t total_number_ = 0;
};

dict uniform_sample(const std::vector<dict>& input);
}  // namespace hami