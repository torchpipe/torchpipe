#include "dict.hpp"
#include <torch/torch.h>

namespace ipipe {

struct HWCTensorWrapper {
 public:
  HWCTensorWrapper(dict input_dict);
  HWCTensorWrapper(dict input_dict, int target_h, int target_w, bool set_zero = false);
  HWCTensorWrapper(dict input_dict, int top, int bottom, int left, int right);
  void finalize();
  ~HWCTensorWrapper() = default;

  torch::Tensor input_tensor;
  torch::Tensor output_tensor;

 private:
  bool is_hwc_input{false};
  dict input;
};

}  // namespace ipipe
