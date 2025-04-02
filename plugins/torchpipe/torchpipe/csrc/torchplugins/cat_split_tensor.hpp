#pragma once

#include <unordered_set>
#include <string>
#include <hami/extension.hpp>
#include "helper/net_info.hpp"

using hami::dict;

namespace torchpipe {

class CatTensor : public hami::BackendMax {
 private:
  void impl_forward(const std::vector<dict>& input_output) override;
};

class FixTensor : public hami::BackendMax {
 private:
  void impl_init(
      const std::unordered_map<std::string, std::string>& config_param,
      const dict& kwargs) override;
  void impl_forward(const std::vector<dict>& input_output) override;

 private:
  std::shared_ptr<NetIOInfos> net_shapes_;
};

class SplitTensor : public hami::Backend {
 private:
  void impl_init(
      const std::unordered_map<std::string, std::string>& config_param,
      const dict& kwargs) override;
  void impl_forward(const std::vector<dict>& input_output) override;

 private:
  std::shared_ptr<std::vector<NetIOInfo>> net_shapes_;
};

class ArgMaxTensor : public hami::Backend {
 private:
  void impl_init(
      const std::unordered_map<std::string, std::string>& config_param,
      const dict& kwargs) override;
  void impl_forward(const std::vector<dict>& io) override;
};

class SoftmaxArgMaxTensor : public hami::Backend {
 private:
  void impl_init(
      const std::unordered_map<std::string, std::string>& config_param,
      const dict& kwargs) override {}
  void impl_forward(const std::vector<dict>& io) override;
};

} // namespace torchpipe