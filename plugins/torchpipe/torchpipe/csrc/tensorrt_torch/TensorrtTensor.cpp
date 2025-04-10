#include <string>
#include "tensorrt_torch/TensorrtTensor.hpp"
#include "hami/builtin/generate_backend.hpp"
namespace torchpipe {

constexpr auto INIT_STR =
    "ModelLoadder[(.onnx)Onnx2Tensorrt,(.trt)LoadTensorrtEngine], "
    "TensorrtInferTensor";
constexpr auto FORWARD_STR =
    "CatSplit[S_v0[FixTensor,CatTensor],S_v0[ContiguousTensor,"
    "Reflect[pre_processor,Identity],Forward[TensorrtInferTensor],Reflect[post_processor,Identity]],"
    "SplitTensor]";
const auto BACKEND_STR =
    std::string() + "IoCV0[" + INIT_STR + "; " + FORWARD_STR + "]";

HAMI_GENERATE_BACKEND(TensorrtTensor, BACKEND_STR, "");

} // namespace torchpipe