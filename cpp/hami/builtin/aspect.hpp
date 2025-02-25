#pragma once

#include <string>
#include <vector>

namespace hami {

class Aspect : public Container {
   public:
    virtual void post_init(const std::unordered_map<std::string, std::string>&,
                           const dict&) override final;

    /**
     * @brief  select a sub-backend.
     */
    virtual void forward(const std::vector<dict>&) override;

    void inject_dependency(Backend* dependency) override final {
        base_dependencies_.front()->inject_dependency(dependency);
    }

   private:
    virtual std::pair<size_t, size_t> update_min_max(
        const std::vector<Backend*>& depends) override;

   private:
};
}  // namespace hami