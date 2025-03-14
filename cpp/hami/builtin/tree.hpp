#pragma once

#include <string>
#include <vector>

namespace hami {
class Tree final : public Container {
   public:
    void post_init(const std::unordered_map<std::string, std::string>& config,
                   const dict& kwargs) override final;
    void impl_forward(const std::vector<dict>&) override;
};

}  // namespace hami