#include <string>
#include "tensorrt_torch/TensorrtTensor.hpp"
#include "hami/builtin/generate_backend.hpp"
namespace torchpipe {

constexpr auto INIT_STR =
    "ModelLoadder[(.onnx)Onnx2Tensorrt,(.trt)LoadTensorrtEngine], "
    "TensorrtInferTensor";
constexpr auto FORWARD_STR =
    "CatSplit[S[GpuTensor,CatTensor],S[ContiguousTensor,"
    "Forward[TensorrtInferTensor],ProxyFromParam[post_processor]],"
    "SplitTensor]";
const auto BACKEND_STR =
    std::string() + "IoC[" + INIT_STR + "; " + FORWARD_STR + "]";

HAMI_GENERATE_BACKEND(TensorrtTensor, BACKEND_STR, "");

}  // namespace torchpipe