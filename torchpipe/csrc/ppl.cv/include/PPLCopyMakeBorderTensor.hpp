#include "Backend.hpp"
#include "params.hpp"
namespace ipipe {
class PPLCopyMakeBorderTensor : public SingleBackend {
 public:
  bool init(const std::unordered_map<std::string, std::string>& config, dict dict_config) override;
  void forward(dict input_dict) override;

 private:
  std::unique_ptr<Params> params_;
  int top_;
  int bottom_;
  int left_;
  int right_;
};
}  // namespace ipipe