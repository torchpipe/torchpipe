#include "Backend.hpp"
#include "params.hpp"
namespace ipipe {
class PPLResizeTensor : public SingleBackend {
 public:
  bool init(const std::unordered_map<std::string, std::string>& config, dict dict_config) override;
  void forward(dict input_dict) override;

 private:
  std::unique_ptr<Params> params_;
  int resize_h_;
  int resize_w_;
};
}  // namespace ipipe