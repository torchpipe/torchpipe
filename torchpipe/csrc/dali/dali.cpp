#ifdef WITH_DALI
#include "dali/c_api.h"
#include "Backend.hpp"
#include "params.hpp"
namespace ipipe {
class Dali : public Backend {
  virtual bool init(const std::unordered_map<std::string, std::string>& config, dict dict_config) {
    params_ = std::unique_ptr<Params>(new Params({}, {"model"}, {}, {}));
    if (!params_->init(config_param)) return false;
    return true;
  };
  virtual void forward(const std::vector<dict>& input_dicts) {}

 private:
  std::unique_ptr<Params> params_;
};
}  // namespace ipipe

#endif