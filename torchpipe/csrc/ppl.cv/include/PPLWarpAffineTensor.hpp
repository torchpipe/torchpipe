#include "Backend.hpp"
#include "params.hpp"
namespace ipipe {
class PPLWarpAffineTensor : public SingleBackend {
 public:
  bool init(const std::unordered_map<std::string, std::string>& config, dict dict_config) override;
  void forward(dict input_dict) override;

 private:
  std::unique_ptr<Params> params_;
  int target_h_;
  int target_w_;
};
}  // namespace ipipe