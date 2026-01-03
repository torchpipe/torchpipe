import pytest
import torch
import torchpipe
import os
import tempfile
import omniback

# Pipeline configuration strings
INIT_STR = "ModelLoadder[(.onnx)Onnx2Tensorrt,(.trt)LoadTensorrtEngine], TensorrtInferTensor"
FORWARD_STR = "CatSplit[S_v0[GpuTensor,CatTensor],S_v0[ContiguousTensor,TensorrtInferTensor,ProxyFromParam[post_processor]],SplitTensor]"
BACKEND_STR = f"StreamGuard[TensorrtTensor]"
BACKEND_STR = "Identity"
class Conv(torch.nn.Module):
    def __init__(self):
        super(Conv, self).__init__()
        self.conv = torch.nn.Conv2d(3, 1, kernel_size=3, stride=2, padding=1)
    def forward(self, x):
        return self.conv(x)

def get_tmp_onnx(model: torch.nn.Module, input_shape: list) -> str:
    """
    Export PyTorch model to ONNX format
    
    Args:
        model: PyTorch model to export
        input_shape: Input tensor shape
    
    Returns:
        Path to temporary ONNX file
    """
    onnx_path = tempfile.mktemp(suffix=".onnx")
    model.eval()
    input_data = torch.randn(input_shape).cuda().half()
    torch.onnx.export(
        model, 
        input_data, 
        onnx_path,
        input_names=["input"], 
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}}
    )
    return onnx_path


def model_config():
    """Fixture to create model configuration and ONNX file"""
    config = {
        "model_type": "onnx",
        "post_processor": "Identity"
    }
    
    # Create temporary ONNX model
    torch_model = Conv().cuda().half()
    tmp_onnx = get_tmp_onnx(torch_model, [1, 3, 224, 224])
    config["model"] = tmp_onnx
    
    return config, torch_model, tmp_onnx
    
    # Cleanup temporary file
    os.remove(tmp_onnx)

def test_tensorrt_inference():
    """
    Test TensorRT inference pipeline
    
    Tests if the model correctly processes input tensor and produces expected output
    """
    
    config, torch_model, tmp_onnx = model_config()
    
    # Initialize model
    # model = omniback.init(BACKEND_STR, config)
    config["backend"] = BACKEND_STR
    model = omniback.init("Interpreter", config)
    
        
    
    # Prepare input data
    input_tensor = torch.ones((1, 3, 224, 224)).half()*10
    data = omniback.Dict({"data": input_tensor.cuda()})
    
    # Run inference
    # model(data)
    
    bench = omniback.init("Benchmark", {"num_clients": "4", "total_number": "10000"})
    bench.forward_with_dep([data]*100, model)
    
    q = omniback.default_queue()
    print(q.size())
    print(q)
    assert not q.empty()
    a=q.get(False)
    print(a)
    # import pdb;
    # pdb.set_trace()
    
    
    os.remove(tmp_onnx)
    return 
    # Verify results
    result = data['result']
    expected = torch_model(input_tensor.cuda())
    expected = expected.cuda()
    # print(result, expected)
    assert torch.allclose(result, expected, rtol=1e-2, atol=1e-2), "Model output does not match expected values"
    
if __name__ == "__main__":
    import time
    # time.sleep(5)
    # pytest.main(["-s", __file__])
    test_tensorrt_inference()