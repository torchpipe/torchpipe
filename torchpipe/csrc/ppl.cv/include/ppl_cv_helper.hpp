#include "dict.hpp"
#include <ATen/ATen.h>

namespace ipipe {

struct HWCTensorWrapper {
 public:
  HWCTensorWrapper(dict input_dict);
  HWCTensorWrapper(dict input_dict, int target_h, int target_w);
  HWCTensorWrapper(dict input_dict, int top, int bottom, int left, int right);
  void finalize();
  ~HWCTensorWrapper()=default;

  at::Tensor input_tensor;
  at::Tensor output_tensor;

 private:
  bool is_hwc_input{false};
  dict input;
};

}  // namespace ipipe
