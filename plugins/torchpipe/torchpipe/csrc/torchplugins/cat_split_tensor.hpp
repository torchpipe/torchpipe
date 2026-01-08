#pragma once

#include <omniback/extension.hpp>
#include <string>
#include <unordered_set>
#include "helper/net_info.hpp"

using om::dict;

namespace torchpipe {

class CatTensor : public om::BackendMax {
 private:
  void impl_init(
      const std::unordered_map<std::string, std::string>& config_param,
      const dict& kwargs) override;
  void impl_forward(const std::vector<dict>& input_output) override;

 private:
  std::optional<int> index_selector_;
};

class FixTensor : public om::BackendMax {
 private:
  void impl_init(
      const std::unordered_map<std::string, std::string>& config_param,
      const dict& kwargs) override;
  void impl_forward(const std::vector<dict>& input_output) override;

 private:
  std::shared_ptr<NetIOInfos> net_shapes_;
};

class SplitTensor : public om::Backend {
 private:
  void impl_init(
      const std::unordered_map<std::string, std::string>& config_param,
      const dict& kwargs) override;
  void impl_forward(const std::vector<dict>& input_output) override;

 private:
  std::shared_ptr<std::vector<NetIOInfo>> net_shapes_;
};

class ArgMaxTensor : public om::Backend {
 private:
  void impl_init(
      const std::unordered_map<std::string, std::string>& config_param,
      const dict& kwargs) override;
  void impl_forward(const std::vector<dict>& io) override;
};

class SoftmaxArgMaxTensor : public om::Backend {
 private:
  void impl_init(
      const std::unordered_map<std::string, std::string>& config_param,
      const dict& kwargs) override {}
  void impl_forward(const std::vector<dict>& ios) override;
};

} // namespace torchpipe