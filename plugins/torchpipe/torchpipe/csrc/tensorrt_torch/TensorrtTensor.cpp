#include "tensorrt_torch/TensorrtTensor.hpp"
#include <string>
#include "omniback/builtin/generate_backend.hpp"
namespace torchpipe {

constexpr auto INIT_STR =
    "ModelLoadder[(.onnx)Onnx2Tensorrt,(.trt)LoadTensorrtEngine], "
    "TensorrtInferTensor";
constexpr auto FORWARD_STR =
    "CatSplit[S[FixTensor,CatTensor],S[ContiguousTensor,"
    "Reflect[pre_processor,Identity],Forward[TensorrtInferTensor],Reflect[post_processor,Identity]],"
    "SplitTensor]";
const auto BACKEND_STR =
    std::string() + "IoCV0[" + INIT_STR + "; " + FORWARD_STR + "]";

OMNI_GENERATE_BACKEND(TensorrtTensor, BACKEND_STR, "");

// IoC[ModelLoadder[(.onnx)Onnx2Tensorrt,(.trt)LoadTensorrtEngine],TensorrtInferTensor;
// CatSplit[S[FixTensor,CatTensor],S[ContiguousTensor,Reflect[pre_processor,Identity],Forward[TensorrtInferTensor],Reflect[post_processor,Identity]],SplitTensor]]]
} // namespace torchpipe