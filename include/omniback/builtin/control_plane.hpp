#pragma once

#include <string>
#include <vector>

#include "omniback/core/backend.hpp"

namespace omniback {
class ControlPlane : public Backend {
 private:
  void impl_init(
      const std::unordered_map<std::string, std::string>& params,
      const dict& options) override final;

  virtual void impl_custom_init(
      const std::unordered_map<std::string, std::string>& params,
      const dict& options) = 0;

  // Default class name if the instance is not create via reflection.
  //   virtual std::string default_cls_name() const { return "ControlPlane"; }

  virtual void update_min_max() = 0;

  [[nodiscard]] size_t impl_max() const override final {
    return max_;
  }
  [[nodiscard]] size_t impl_min() const override final {
    return min_;
  }

 protected:
  // for T[(a1, a2,b=2)A(x)[B],  X; C]
  std::vector<std::pair<
      std::vector<std::string>,
      std::unordered_map<std::string, std::string>>>
      prefix_args_kwargs_; // *[<'a1','a2',{b:2}>, <>, <>]
  std::vector<std::string> backend_cfgs_; // *[`A(x)[B]`, `X`, 'C']

  std::vector<char>
      delimiters_; //[',',';'] delimiters_.size() +1 == backend_cfgs_.size()
  std::vector<std::string> main_backends_; // [A, X, C]

  size_t min_{1};
  size_t max_{std::numeric_limits<std::size_t>::max()};
};

} // namespace omniback