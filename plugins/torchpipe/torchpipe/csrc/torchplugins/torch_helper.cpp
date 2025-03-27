// #include "torchplugins/torch_helper.hpp"
// #include "hami/core/reflect.h"
// namespace torchpipe {
// void get_output(
//     const std::vector<torch::Tensor>& in,
//     const std::vector<torch::Tensor>& out,
//     const std::string& instance_name) {
//   auto* pbackend = HAMI_INSTANCE_GET(hami::Backend, instance_name);
//   HAMI_ASSERT(pbackend);
//   auto io = hami::make_dict();
//   (*io)[TASK_DATA_KEY] = in;
//   (*io)[TASK_OUTPUT_KEY] = out;
//   pbackend->forward(io);
// }

// } // namespace torchpipe