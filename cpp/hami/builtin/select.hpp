#pragma once

#include <string>
#include <vector>

namespace hami {

class Select : public Container {
   public:
    virtual void post_init(const std::unordered_map<std::string, std::string>&,
                           const dict&) override final;

    /**
     * @brief  select a sub-backend.
     */
    virtual void impl_forward(const std::vector<dict>&) override;

    virtual std::function<size_t(const dict&)> select_impl() const = 0;

   private:
    virtual std::pair<size_t, size_t> update_min_max(
        const std::vector<Backend*>& depends) override;

   private:
    std::function<size_t(const dict&)> select_;
};
}  // namespace hami