import pytest
# import torch
# import torchpipe
import os
import tempfile
import hami

# Pipeline configuration strings
INIT_STR = "ModelLoadder[(.onnx)Onnx2Tensorrt,(.trt)LoadTensorrtEngine], TensorrtInferTensor"
FORWARD_STR = "CatSplit[S[GpuTensor,CatTensor],S[ContiguousTensor,TensorrtInferTensor,ProxyFromParam[post_processor]],SplitTensor]"
BACKEND_STR = f"StreamGuard[TensorrtTensor]"
BACKEND_STR = "Identity"
  

def model_config():
    """Fixture to create model configuration and ONNX file"""
    config = {
        "model_type": "onnx",
        "post_processor": "Identity"
    } 
    
    return config, None, None
    
    # Cleanup temporary file
    os.remove(tmp_onnx)

def test_tensorrt_inference():
    """
    Test TensorRT inference pipeline
    
    Tests if the model correctly processes input tensor and produces expected output
    """
    
    config, torch_model, tmp_onnx = model_config()
    
    # Initialize model
    # model = hami.init(BACKEND_STR, config)
    config["backend"] = BACKEND_STR
    model = hami.init("Interpreter", config)
    
        
    
    # Prepare input data
    input_tensor = 1
    data = {"data": input_tensor}
    
    # Run inference
    # model(data)
    
    bench = hami.init("Benchmark", {"num_clients": "4", "total_number": "10000"})
    bench.forward([data]*100, model)
    
    q = hami._C.default_queue()
    print(q.size())
    print(q)
    assert not q.empty()
    a=q.get()
    print(a)
    # import pdb;
    # pdb.set_trace()
    
    
    return 
   
if __name__ == "__main__":
    import time
    # time.sleep(5)
    # pytest.main(["-s", __file__])
    test_tensorrt_inference()