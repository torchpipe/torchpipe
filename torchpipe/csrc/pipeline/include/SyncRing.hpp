#pragma once

#include "Backend.hpp"
#include "params.hpp"

namespace ipipe {
class SyncRing : public Backend {
 public:
  bool init(const std::unordered_map<std::string, std::string>& config, dict dict_config) override;
  void forward(const std::vector<dict>& input_dicts) override;

 private:
  // std::set<std::string> ring_;
  std::unique_ptr<Params> params_;
  std::unique_ptr<Backend> backend_;
};
}  // namespace ipipe
