import pytest
import torchpipe

import torch
import os
import tempfile
import omniback

# Pipeline configuration strings
INIT_STR = "ModelLoadder[(.onnx)Onnx2Tensorrt,(.trt)LoadTensorrtEngine], TensorrtInferTensor"
FORWARD_STR = "CatSplit[S_v0[GpuTensor,CatTensor],S_v0[ContiguousTensor,Forward[TensorrtInferTensor],ProxyFromParam[post_processor]],SplitTensor]"
BACKEND_STR = f"IoCV0[{INIT_STR}; {FORWARD_STR}]"

class Identity(torch.nn.Module):
    """Simple identity model that multiplies input by 2"""
    def forward(self, x):
        return x * 2

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
    input_data = torch.randn(input_shape)
    torch.onnx.export(
        model, 
        input_data, 
        onnx_path,
        input_names=["input"], 
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}}
    )
    return onnx_path

@pytest.fixture
def model_config():
    """Fixture to create model configuration and ONNX file"""
    config = {
        "model_type": "onnx",
        "post_processor": "Identity"
    }
    
    # Create temporary ONNX model
    tmp_onnx = get_tmp_onnx(Identity(), [1, 3, 224, 224])
    config["model"] = tmp_onnx
    
    yield config, tmp_onnx
    
    # Cleanup temporary file
    os.remove(tmp_onnx)

def test_tensorrt_inference(model_config):
    """
    Test TensorRT inference pipeline
    
    Tests if the model correctly processes input tensor and produces expected output
    """
    config, _ = model_config
    
    # Initialize model
    model = omniback.init(BACKEND_STR, config)
    
    # Prepare input data
    input_tensor = torch.ones((1, 3, 224, 224))
    data = {"data": input_tensor}
    
    # Run inference
    model(data)
    
    # Verify results
    result = data['result']
    print(result)
    if not isinstance(result, torch.Tensor):
        result = torch.from_dlpack(result)
        
    expected = input_tensor * 2
    expected = expected.cuda()
    
    assert torch.allclose(result, expected), "Model output does not match expected values"