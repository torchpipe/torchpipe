

#pragma once

#include <string>
#include <vector>
#include "omniback/builtin/basic_backends.hpp"
namespace omniback {
class CatSplit final : public Container {
 public:
  virtual void post_init(
      const std::unordered_map<std::string, std::string>&,
      const dict&) override;

  virtual void impl_forward(const std::vector<dict>&) override;

 private:
  virtual std::vector<size_t> set_init_order(size_t max_range) const override;

  std::pair<size_t, size_t> update_min_max(
      const std::vector<Backend*>& depends) override;
};

} // namespace omniback