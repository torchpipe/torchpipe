



import torchpipe

init = "ModelLoadder[(.onnx)Onnx2Tensorrt,(.trt)LoadTensorrtEngine], TensorrtInferTensor"
forward = "CatSplit[S[GpuTensor,CatTensor],S[ContiguousTensor,TensorrtInferTensor,LaunchFromParam[post_processor]],SplitTensor]"
backend_str = f"""
IoC[{init}; {forward}]
"""


import time
# time.sleep(5)

print(f"backend_str={backend_str}")
import hami
hami.init(backend_str)
hami.init(backend_str, {"post_processor": "Identity"})