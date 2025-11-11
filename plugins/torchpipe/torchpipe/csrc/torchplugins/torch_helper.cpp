// #include "torchplugins/torch_helper.hpp"
// #include "omniback/core/reflect.h"
// namespace torchpipe {
// void get_output(
//     const std::vector<torch::Tensor>& in,
//     const std::vector<torch::Tensor>& out,
//     const std::string& instance_name) {
//   auto* pbackend = OMNI_INSTANCE_GET(omniback::Backend, instance_name);
//   OMNI_ASSERT(pbackend);
//   auto io = omniback::make_dict();
//   (*io)[TASK_DATA_KEY] = in;
//   (*io)[TASK_OUTPUT_KEY] = out;
//   pbackend->forward(io);
// }

// } // namespace torchpipe