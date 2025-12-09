#pragma once

#include <string>
#include <vector>
#include "omniback/builtin/basic_backends.hpp"

namespace omniback {

class Aspect : public Container {
 public:
  virtual void post_init(
      const std::unordered_map<std::string, std::string>&,
      const dict&) override final;

  /**
   * @brief  select a sub-backend.
   */
  virtual void impl_forward(const std::vector<dict>&) override;

  void impl_inject_dependency(Backend* dependency) override final {
    base_dependencies_.front()->inject_dependency(dependency);
  }

 private:
  virtual std::pair<size_t, size_t> update_min_max(
      const std::vector<Backend*>& depends) override;

 private:
};

} // namespace omniback
